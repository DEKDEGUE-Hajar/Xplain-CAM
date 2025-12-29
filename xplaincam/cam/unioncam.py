from __future__ import annotations


import torch
import torch.nn.functional as F
import numpy as np
from .basecam import BaseCAM
from .scorecam import ScoreCAM


class UnionCAM(BaseCAM):

    """
    Union-CAM: an ensemble CAM that combines gradient-based and region-based maps 
    by denoising the gradient-based map, linearly combining it with the region-based map weighted by their scores, 
    and choosing as the final map the one with the higher class score between the combined map and the region-based map.

    Reference:
        Hu et al., "UnionCAM: enhancing CNN interpretability through denoising, weighted fusion, and selective high                quality class activation mapping",
        https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1490198/full
    """



    def __init__(self, model, target_layer=None, input_shape=(3, 224, 224), device=None):
        super().__init__(model, target_layer, input_shape)
        self.device = device if device is not None else next(model.parameters()).device
        self.scorecam = ScoreCAM(model, target_layer=target_layer)

    def _generate_denoised_cam(self, x, class_idx=None, theta=10):
        """Generate denoised CAM component"""
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[0, class_idx]

        self.model.zero_grad()
        score.backward()

        # Use activations and gradients from BaseCAM
        A = self.activations['value'][0]  # [C, H, W]
        W = self.gradients['value'][0].clone()  # [C, H, W]

        # Channel-wise denoising
        for c in range(W.shape[0]):
            threshold_c = np.percentile(W[c].cpu().numpy(), theta)
            mask = W[c] < threshold_c
            W[c][mask] = 0

        relu_W = F.relu(W)
        indicator = (W > 0).float()
        sum_weights = torch.sum(W * indicator, dim=(1, 2), keepdim=True) + 1e-8
        alpha = indicator / sum_weights

        # Weighted sum over channels
        L_denoising = torch.sum(alpha * relu_W * A, dim=0)  # [H, W]

        # Upsample to input size
        L_denoising = L_denoising.unsqueeze(0).unsqueeze(0)  # add batch & channel dims
        L_denoising = F.interpolate(
            L_denoising, size=(x.shape[2], x.shape[3]),
            mode='bilinear', align_corners=False
        )

        # Normalize
        cam_min, cam_max = L_denoising.min(), L_denoising.max()
        L_denoising = (L_denoising - cam_min) / (cam_max - cam_min + 1e-8)

        return L_denoising, class_idx

    def normalize_heatmap(self, heatmap):
        """Normalize heatmap to [0,1] range (tensor)"""
        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap).float().to(self.device)
        cam_min, cam_max = heatmap.min(), heatmap.max()
        heatmap = (heatmap - cam_min) / (cam_max - cam_min + 1e-8)
        return heatmap

    def forward(self, x, class_idx=None, theta=10):
        """Generate Union-CAM map"""
        L_denoising, predicted_class = self._generate_denoised_cam(x, class_idx, theta)
        L_region = self.scorecam(x, class_idx)

        # Baseline black image scores
        Ib = torch.zeros_like(x).to(self.device)
        score_black = F.softmax(self.model(Ib), dim=-1)[0, predicted_class]

        # Scores for components
        score_denoising = F.softmax(self.model(x * L_denoising), dim=-1)[0, predicted_class]
        score_region = F.softmax(self.model(x * L_region), dim=-1)[0, predicted_class]

        beta_denoising = score_denoising - score_black
        beta_region = score_region - score_black

        # Weighted combination
        L_de_region = beta_denoising * L_denoising + beta_region * L_region
        L_de_region = self.normalize_heatmap(L_de_region)

        score_de_region = F.softmax(self.model(x * L_de_region), dim=-1)[0, predicted_class]
        beta_de_region = score_de_region - score_black

        # Select final CAM
        if beta_de_region > beta_region:
            union_cam = L_de_region 
            selected_method = "De-Region"
        else:
            union_cam = L_region 
            selected_method = "Region"

        return union_cam

    def __call__(self, x, class_idx=None, theta=10):
        return self.forward(x, class_idx, theta)
