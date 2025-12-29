from __future__ import annotations

import torch
import torch.nn.functional as F
from .scorecam import ScoreCAM


class SSCAM(ScoreCAM):
    
    """
    SS-CAM: extends Score-CAM by applying smoothing or random perturbations to each activation map,
    the final map is obtained by averaging the scores across these perturbed maps.

    Reference:
        Wang et al., "SS-CAM: Smoothed Score-CAM for Sharper Visual Feature Localization",
        https://arxiv.org/pdf/2006.14255
    """

    def __init__(
        self,
        model,
        target_layer=None,
        num_samples: int = 35,
        std: float = 2.0,
        batch_size: int = 32,
    ):
        """
        Args:
            model: Trained neural network model.
            target_layer: Target convolutional layer.
            num_samples: Number of noise samples per activation.
            std: Standard deviation of Gaussian noise.
            batch_size: Number of activation maps processed at once.
        """
        super().__init__(model, target_layer, batch_size=batch_size)
        self.num_samples = num_samples
        self.std = std
        
    def forward(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """
        Compute SS-CAM saliency map.

        Args:
            x: Input tensor of shape (B, C, H, W).
            class_idx: Target class index. Uses predicted class if None.
            retain_graph: Kept for API compatibility.

        Returns:
            Normalized saliency map of shape (B, 1, H, W).
        """
        device = x.device
        b, _, h, w = x.size()
        
        # Get baseline scores
        with torch.no_grad():
            baseline_logits = self.model(x)
            if class_idx is None:
                class_idx = baseline_logits.argmax(dim=1)
            else:
                class_idx = torch.tensor([class_idx] * b, device=device)
            
        # Get activations
        activations = self.activations["value"].to(device)
        _, k, act_h, act_w = activations.size()
        
        # Upsample activations to input size
        upsampled_acts = F.interpolate(
            activations, size=(h, w), mode="bilinear", align_corners=False
        )
        
        # Initialize weights
        weights = torch.zeros(b, k, device=device)
        
        # Process each activation map
        for i in range(0, k, self.batch_size):
            batch_end = min(i + self.batch_size, k)
            batch_k = batch_end - i
            
            # Get batch of upsampled activations
            batch_acts = upsampled_acts[:, i:batch_end]
            
            # Process each activation channel
            for j in range(batch_k):
                act = batch_acts[:, j:j+1]
                score_sum = torch.zeros(b, device=device)
                
                # Generate multiple noisy samples
                for _ in range(self.num_samples):
                    # Add Gaussian noise before normalization
                    noise = torch.randn_like(act) * self.std
                    noisy_act = act + noise
                    
                    # Normalize with noise
                    act_min = noisy_act.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
                    act_max = noisy_act.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
                    normalized_act = (noisy_act - act_min) / (act_max - act_min + 1e-8)
                    
                    # Create masked input
                    masked_input = x * normalized_act
                    
                    # Get model output and subtract baseline
                    with torch.no_grad():
                        output_logits = self.model(masked_input)
                        score_diff = output_logits - baseline_logits
                        
                        if len(class_idx) == 1:
                            score_sum += score_diff[:, class_idx[0]]
                        else:
                            score_sum += torch.gather(score_diff, 1, class_idx.unsqueeze(1)).squeeze(1)
                
                # Average over samples and store
                weights[:, i + j] = score_sum / self.num_samples
        
        # Apply softmax to weights
        weights = F.softmax(weights, dim=1)
        
        # Compute CAM
        cam = torch.zeros((b, 1, h, w), device=device)
        for i in range(k):
            weight = weights[:, i].view(b, 1, 1, 1)
            cam += weight * upsampled_acts[:, i:i+1]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam_min = cam.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        cam_max = cam.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam
    
    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)