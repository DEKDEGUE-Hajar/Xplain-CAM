from __future__ import annotations

import torch
import torch.nn.functional as F
import cv2
import numpy as np

class AvgDropConf:
    """
    Compute Average Drop and Increase for a given model and heatmap.
    Computes metrics immediately upon initialization.
    """

    def __init__(self, model, heatmap, input_tensor, class_idx=None, alpha=1.0):
        """
        Args:
            model: the torch model
            heatmap: CAM heatmap (Tensor or np.array)
            input_tensor: normalized model input (1,3,H,W)
            class_idx: class index to compute drop for (optional)
            alpha: optional power applied to heatmap (default 1.0)
        """
        self.model = model
        self.heatmap = heatmap
        self.input_tensor = input_tensor
        self.alpha = alpha

        self.avg_drop, self.avg_increase = self._compute(class_idx)

    def _compute(self, class_idx):
        device = self.input_tensor.device

        # Original prediction
        with torch.no_grad():
            output = self.model(self.input_tensor)
            probs = F.softmax(output, dim=1)
            if class_idx is None:
                class_idx = torch.argmax(probs, dim=1).item()
            original_conf = probs[0, class_idx].item()

            # Prepare heatmap
            if isinstance(self.heatmap, torch.Tensor):
                heatmap = self.heatmap.squeeze().cpu().numpy()
            else:
                heatmap = self.heatmap

            _, _, H, W = self.input_tensor.shape
            heatmap = cv2.resize(heatmap, (W, H))
            heatmap = torch.from_numpy(heatmap).float().to(device)
            heatmap = heatmap.unsqueeze(0).unsqueeze(0) ** self.alpha
            heatmap = heatmap.repeat(1, 3, 1, 1)

            # Masked input
            masked_input = self.input_tensor * heatmap

            new_output = self.model(masked_input)
            new_probs = F.softmax(new_output, dim=1)
            new_conf = new_probs[0, class_idx].item()

            drop = max(0, original_conf - new_conf) / (original_conf + 1e-8)
            increase = 1 if new_conf > original_conf else 0

        return drop, increase

    def summary(self):
        return {"avg_drop": float(self.avg_drop), "avg_increase": float(self.avg_increase)}
