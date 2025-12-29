from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from .basecam import BaseCAM
from .scorecam import ScoreCAM
from .gradcam import GradCAM


class FusionCAM(BaseCAM):
    
    """
    Fusion-CAM: an ensemble CAM that combines gradient-based and region-based maps 
    by denoising the gradient-based map, linearly combining it with the region-based map weighted by their scores, 
    and applying a similarity-based fusion to produce the final map.
    
    """


    def __init__(
        self,
        model,
        target_layer,
        input_shape=(3, 224, 224),
        grad_cam=GradCAM,
        region_cam=ScoreCAM,
        device=None,
        **cam_kwargs,
    ):
        """
        Args:
            model: Trained neural network model.
            target_layer: Target convolutional layer.
            input_shape: Expected input shape (C, H, W).
            grad_cam: Gradient-based CAM class.
            region_cam: Region-based CAM class.
            device: Torch device. Inferred if None.
        """
        super().__init__(model, target_layer, input_shape)
        self.device = device or next(model.parameters()).device

        self.grad_cam = grad_cam(
            model, target_layer=target_layer, **cam_kwargs
        )
        self.region_cam = region_cam(
            model, target_layer=target_layer, **cam_kwargs
        )

    def normalize_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Normalize heatmap to [0, 1].
        """
        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap).float().to(self.device)

        cam_min, cam_max = heatmap.min(), heatmap.max()
        return (heatmap - cam_min) / (cam_max - cam_min + 1e-8)
    
    def _generate_gradcam_denoised(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        theta: float = 10,
    ) -> torch.Tensor:
        """
        Generate denoised Grad-CAM via percentile thresholding.

        Args:
            x: Input tensor of shape (1, C, H, W).
            class_idx: Target class index.
            theta: Percentile threshold (0 disables denoising).

        Returns:
            Denoised Grad-CAM heatmap.
        """
        cam = self.grad_cam(x, class_idx)

        threshold = np.percentile(
            cam.detach().cpu().numpy(), theta
        )

        denoised_cam = torch.where(cam > threshold, cam, torch.zeros_like(cam),)
        
        return  denoised_cam



    def forward(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        theta: float = 10,
    ) -> torch.Tensor:
        """
        Compute Fusion-CAM.

        Args:
            x: Input tensor of shape (1, C, H, W).
            class_idx: Target class index.
            theta: Percentile threshold for Grad-CAM denoising.

        Returns:
            Fusion-CAM heatmap.
        """
        logit = self.model(x)

        if class_idx is None:
            predicted_class = logit.argmax(dim=1)
        else:
            predicted_class = torch.tensor(
                [class_idx], device=self.device
            )

        # Component CAMs
        L_denoising = self._generate_gradcam_denoised(x, class_idx, theta)
        L_region = self.region_cam(x, class_idx)

        # Baseline (black image)
        baseline = torch.zeros_like(x)
        score_black = F.softmax(
            self.model(baseline), dim=-1
        )[0, predicted_class]

        # Component scores
        score_denoising = F.softmax(
            self.model(x * L_denoising), dim=-1
        )[0, predicted_class]

        score_region = F.softmax(
            self.model(x * L_region), dim=-1
        )[0, predicted_class]

        beta_denoising = score_denoising - score_black
        beta_region = score_region - score_black

        # Weighted fusion
        L_fused = (beta_denoising * L_denoising + beta_region * L_region)
        L_fused = self.normalize_heatmap(L_fused)

        score_fused = F.softmax(self.model(x * L_fused), dim=-1)[0, predicted_class]

        beta_fused = score_fused - score_black

        # Similarity-based fusion
        L1 = L_fused * beta_fused
        L2 = L_region * beta_region
    
        diff = torch.abs(L1 - L2)
        sim = torch.clamp(1.0 - diff, 0.0, 1.0)

        fusion_cam = (sim * torch.max(L1, L2) + (1.0 - sim) * 0.5 * (L1 + L2))

        return self.normalize_heatmap(fusion_cam)

    def __call__(self, x, class_idx=None, theta=10):
        return self.forward(x, class_idx, theta)
