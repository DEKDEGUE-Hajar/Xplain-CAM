from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from .basecam import BaseCAM

class ScoreCAM(BaseCAM):

    """
    Score-CAM: Measures the increase in class score when each activation map 
    is masked and used as input, then linearly combines the maps based on these scores.

    Reference:
        Wang et al., "Score-CAM: Score-Weighted Visual Explanations for CNNs",
        https://arxiv.org/pdf/1910.01279.pdf

    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: str | torch.nn.Module | None = None,
        input_shape: tuple[int, int, int] = (3, 224, 224),
        batch_size: int = 32
    ):
        """
        Initialize ScoreCAM.

        Args:
            model: CNN model to explain.
            target_layer: Layer to extract activations. If None, last conv layer is used.
            input_shape: Input shape (C, H, W).
            batch_size: Number of masked inputs per batch.
        """
        super().__init__(model, target_layer, input_shape)
        self.batch_size = batch_size

    def forward(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        retain_graph: bool = False
    ) -> torch.Tensor:
        
        """Compute Score-CAM heatmap.

        Args:
            x: Input tensor (B, C, H, W).
            class_idx: Target class index. Uses predicted class if None.
            retain_graph: Keep computation graph (not needed for Score-CAM).

        Returns:
            Normalized CAM tensor (B, 1, H, W).
        """

        device = x.device
        b, c, h, w = x.size()

        # Get raw model prediction
        logit = self.model(x)
        if class_idx is None:
            predicted_class = logit.argmax(dim=1)
        else:
            predicted_class = torch.LongTensor([class_idx]).to(device)

        # Get activations from target layer
        activations = self.activations['value'].detach().to(device)

        k = activations.size(1)

        # Normalize each activation map to [0,1]
        norm_activations = []
        for i in range(k):
            act = activations[:, i:i+1, :, :]
            act_min = act.view(b, -1).min(dim=1)[0].view(b,1,1,1)
            act_max = act.view(b, -1).max(dim=1)[0].view(b,1,1,1)
            norm = (act - act_min) / (act_max - act_min + 1e-8)
            norm_activations.append(norm)
        norm_activations = torch.cat(norm_activations, dim=1)  # [b, k, Hc, Wc]

        # Initialize saliency map
        score_saliency_map = torch.zeros((b,1,h,w), device=device)

        # Process in batches for efficiency
        for i in range(0, k, self.batch_size):
            batch_acts = norm_activations[:, i:i+self.batch_size, :, :]  # [b, batch_k, Hc, Wc]
            batch_k = batch_acts.size(1)

            # Upsample to input size
            batch_acts_upsampled = F.interpolate(batch_acts, size=(h,w), mode='bilinear', align_corners=False)

            # Skip empty maps
            if (batch_acts_upsampled.max(dim=1)[0] == batch_acts_upsampled.min(dim=1)[0]).all():
                continue

            # Mask input with each activation map
            for j in range(batch_k):
                masked_input = x * batch_acts_upsampled[:, j:j+1, :, :]
                with torch.no_grad():
                    output = self.model(masked_input)
                    output = F.softmax(output, dim=-1)

                if class_idx is None:
                    score = output[:, predicted_class].view(b,1,1,1)
                else:
                    score = output[:, class_idx].view(b,1,1,1)
                score_saliency_map += score * batch_acts_upsampled[:, j:j+1, :, :]

        # ReLU and normalize
        score_saliency_map = F.relu(score_saliency_map)
        cam_min = score_saliency_map.view(b, -1).min(dim=1)[0].view(b,1,1,1)
        cam_max = score_saliency_map.view(b, -1).max(dim=1)[0].view(b,1,1,1)
        score_saliency_map = (score_saliency_map - cam_min) / (cam_max - cam_min + 1e-8)

        return score_saliency_map

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
