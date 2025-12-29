from __future__ import annotations


import torch
import torch.nn.functional as F
from .basecam import BaseCAM


class XGradCAM(BaseCAM):
    """
    XGrad-CAM: Scales each feature map by the ratio of its weighted gradient 
    sum to its total activation, emphasizing neurons proportionally to their influence.

    Reference:
        Fu et al., XGrad-CAM: Explicit Gradient-based Localization
        for Deep Networks. BMVC 2020.
        https://arxiv.org/abs/2004.10528
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer,
        input_shape: tuple[int, int, int] = (3, 224, 224),
    ):
        """
        Args:
            model: CNN model to explain.
            target_layer: Target convolutional layer.
            input_shape: Expected input shape (C, H, W).
        """
        super().__init__(model, target_layer, input_shape)

    def forward(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """
        Compute XGrad-CAM for an input image.

        Args:
            x: Input tensor of shape (B, C, H, W).
            class_idx: Target class index. Uses predicted class if None.
            retain_graph: Whether to retain the computation graph.

        Returns:
            Normalized CAM tensor of shape (B, 1, H, W).
        """
        b, _, h, w = x.size()

        # Forward pass
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
            score = logits.gather(1, class_idx.view(-1, 1)).squeeze()
        else:
            score = logits[:, class_idx].squeeze()

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)

        gradients = self.gradients["value"]      # dY/dA
        activations = self.activations["value"]  # A
        _, k, _, _ = activations.size()

        # XGrad-CAM weights
        weights = (
            (gradients * activations)
            .view(b, k, -1).sum(dim=2)
            / (activations.view(b, k, -1).sum(dim=2) + 1e-8)
        )
        weights = weights.view(b, k, 1, 1)

        # Compute CAM
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
