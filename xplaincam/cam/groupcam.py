from __future__ import annotations

import torch
import torch.nn.functional as F
from kornia.filters.gaussian import gaussian_blur2d

from .basecam import BaseCAM
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

blur = lambda x: gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50.0, 50.0))

def group_cluster(x, group=32, cluster_method='k_means'):
    # x : (torch tensor with shape [1, c, h, w])
    xs = x.detach().cpu()
    b, c, h, w = xs.shape
    xs = xs.reshape(b, c, -1).reshape(b*c, h*w)
    if cluster_method == 'k_means':
        n_cluster = KMeans(n_clusters=group, random_state=0).fit(xs)
    elif cluster_method == 'agglomerate':
        n_cluster = AgglomerativeClustering(n_clusters=group).fit(xs)
    else:
        assert NotImplementedError

    labels = n_cluster.labels_
    del xs
    return labels


def group_sum(x, n=32, cluster_method='k_means'):
    b, c, h, w = x.shape
    group_idx = group_cluster(x, group=n, cluster_method=cluster_method)
    init_masks = [torch.zeros(1, 1, h, w).to(x.device) for _ in range(n)]
    for i in range(c):
        idx = group_idx[i]
        init_masks[idx] += x[:, i, :, :].unsqueeze(1)
    return init_masks



class GroupCAM(BaseCAM):
    """
    Group-CAM: divides feature maps into groups and creates a mask for each group. 
    It measures each mask’s effect on the class score and combines them to produce 
    the final activation map.


    Reference:
        Zhang et al., "Group-CAM: Group Score-Weighted Visual Explanations for Deep Convolutional Networks"
        https://arxiv.org/pdf/2103.13859
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer,
        input_shape: tuple[int, int, int] = (3, 224, 224),
        groups: int = 32,
        cluster_method: str | None = None,
    ):
        """
        Args:
            model: CNN model to explain.
            target_layer: Target convolutional layer.
            input_shape: Expected input shape (C, H, W).
            groups: Number of feature groups.
            cluster_method: Grouping strategy
                (None, "k_means", or "agglomerate").
        """
        super().__init__(model, target_layer, input_shape)
        assert cluster_method in [None, "k_means", "agglomerate"]
        self.groups = groups
        self.cluster = cluster_method

    def forward(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        retain_graph: bool = False,
    ) -> torch.Tensor | None:
        """
        Compute Group-CAM for an input image.

        Args:
            x: Input tensor of shape (1, C, H, W).
            class_idx: Target class index. Uses predicted class if None.
            retain_graph: Whether to retain the computation graph.

        Returns:
            Normalized CAM tensor of shape (1, 1, H, W),
            or None if the map is degenerate.
        """
        b, _, h, w = x.size()
        device = x.device

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
            score = logits.gather(1, class_idx.view(-1, 1)).squeeze()
        else:
            score = logits[:, class_idx].squeeze()
            class_idx = torch.tensor([class_idx], device=device)

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)

        gradients = self.gradients["value"]
        activations = self.activations["value"]
        _, k, _, _ = activations.size()

        # Channel weighting (Grad-CAM style)
        weights = gradients.view(b, k, -1).mean(dim=2).view(b, k, 1, 1)
        activations = weights * activations

        # ---- Grouping ----
        if self.cluster is None:
            grouped = activations.chunk(self.groups, dim=1)
            saliency_map = torch.cat(grouped, dim=0).sum(dim=1, keepdim=True)
        else:
            grouped = group_sum(activations, n=self.groups, cluster_method=self.cluster)
            saliency_map = torch.cat(grouped, dim=0)

        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(
            saliency_map, size=(h, w), mode="bilinear", align_corners=False
        )

        # Normalize per group
        flat = saliency_map.view(self.groups, -1)
        flat_min = flat.min(dim=1, keepdim=True)[0]
        flat_max = flat.max(dim=1, keepdim=True)[0]
        norm_saliency_map = (flat - flat_min) / (flat_max - flat_min + 1e-8)
        norm_saliency_map = norm_saliency_map.view(self.groups, 1, h, w)

        # Group-wise scoring
        with torch.no_grad():
            baseline = F.softmax(self.model(blur(x)), dim=-1)[0, class_idx]
            masked = x * norm_saliency_map + blur(x) * (1 - norm_saliency_map)
            output = F.softmax(self.model(masked), dim=-1)

        score = F.relu(output[:, class_idx] - baseline).view(self.groups, 1, 1, 1)
        cam = torch.sum(saliency_map * score, dim=0, keepdim=True)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_min == cam_max:
            return None

        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
