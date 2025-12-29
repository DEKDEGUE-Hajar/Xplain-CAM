from __future__ import annotations

import torch
import torch.nn.functional as F
from .basecam import BaseCAM


class GradCAMpp(BaseCAM):
    """
    Grad-CAM++: Extends Grad-CAM by using weighted combination of first- and second-order 
    gradients for better localization of multiple occurrences of the target object.

    Reference:
        Chattopadhyay et al., Grad-CAM++: Improved Visual Explanations
        for Deep Convolutional Networks. WACV 2018.
        https://arxiv.org/abs/1710.11063
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer,
        input_shape: tuple[int, int, int] = (3, 224, 224)
    ):
        """
        Args:
            model: CNN model to explain.
            target_layer: Target convolutional layer.
            input_shape: Input shape (C, H, W) for CAM upsampling.
        """
        super().__init__(model, target_layer, input_shape)

    def forward(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        retain_graph: bool = False
    ) -> torch.Tensor:
        """
        Compute Grad-CAM++ for an input tensor.

        Args:
            x: Input tensor of shape (B, C, H, W).
            class_idx: Target class index. Uses predicted class if None.
            retain_graph: Whether to retain computation graph.

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

        gradients = self.gradients["value"]      # dS/dA
        activations = self.activations["value"]  # A
        _, k, u, v = activations.size()

        # Grad-CAM++ weights
        alpha_num = gradients.pow(2)
        alpha_denom = (
            gradients.pow(2).mul(2) +
            activations.mul(gradients.pow(3))
            .view(b, k, -1).sum(dim=2, keepdim=True)
            .view(b, k, 1, 1)
        )
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num / (alpha_denom + 1e-8)

        positive_gradients = F.relu(score.exp().view(b, 1, 1, 1) * gradients)
        weights = (alpha * positive_gradients).view(b, k, -1).sum(dim=2).view(b, k, 1, 1)

        # Compute CAM
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
