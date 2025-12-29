from __future__ import annotations

import torch
import torch.nn.functional as F
from .basecam import BaseCAM


class GradCAM(BaseCAM):
    """
    Grad-CAM: Computes gradients of the target class w.r.t. feature maps, 
    averages them across spatial dimensions, and weights the maps to get 
    the final class activation map.  
    
    Reference:
        Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization. ICCV 2017.
        https://arxiv.org/abs/1610.02391
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: str | torch.nn.Module | None = None,
        input_shape: tuple[int, int, int] = (3, 224, 224)
    ):
        """Initialize GradCAM.

        Args:
            model: CNN model to explain.
            target_layer: Layer to extract activations; if None, last conv layer is used.
            input_shape: Expected input shape (C, H, W) for upsampling CAM.
        """
        super().__init__(model, target_layer, input_shape)

    def forward(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        retain_graph: bool = False
    ) -> torch.Tensor:
        """Compute Grad-CAM heatmap.

        Args:
            x: Input tensor (B, C, H, W).
            class_idx: Target class index. Uses predicted class if None.
            retain_graph: Keep computation graph for multiple backward passes.

        Returns:
            Normalized CAM tensor (B, 1, H, W) in [0,1].
        """
        # Forward pass
        logits = self.model(x)

        # Select target score
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx].squeeze()

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)

        # Get activations and gradients
        activations = self.activations["value"].data      # (B, K, H, W)
        gradients = self.gradients["value"].data          # (B, K, H, W)
        b, k, u, v = activations.size()

        # Compute weights and CAM
        alpha = gradients.view(b, k, -1).mean(dim=2)      # Global average pooling
        weights = alpha.view(b, k, 1, 1)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))

        # Upsample to input size
        target_h, target_w = self.input_shape[1], self.input_shape[2]
        cam = F.interpolate(cam, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # Normalize to [0,1]
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        retain_graph: bool = False
    ) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(x, class_idx, retain_graph)
