from __future__ import annotations

import torch
import torch.nn.functional as F
from .scorecam import ScoreCAM


class ISCAM(ScoreCAM):
    
    """
    IS-CAM: borrows the “integration idea” from IG: instead of looking at the activation map once 
    (like Score-CAM does), it gradually scales the map from zero to full and averages the class-score 
    effects over these steps.

    Reference:
        Naidu et al., IS-CAM: Integrated Score-CAM for axiomatic-based explanations.
        https://arxiv.org/pdf/2010.03023   
   """

    def __init__(
        self,
        model,
        target_layer,
        num_samples: int = 10,
        batch_size: int = 32,
    ):
        """
        Args:
            model: Trained neural network model.
            target_layer: Target convolutional layer.
            num_samples: Number of integration steps.
            batch_size: Batch size for masked forward passes.
        """
        super().__init__(model, target_layer)
        self.num_samples = num_samples
        self.batch_size = batch_size

    def forward(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """
        Compute IS-CAM saliency map.

        Args:
            x: Input tensor of shape (B, C, H, W).
            class_idx: Target class index. Uses predicted class if None.
            retain_graph: Kept for API compatibility.

        Returns:
            Normalized saliency map of shape (B, 1, H, W).
        """
        device = x.device
        b, _, h, w = x.size()

        # Get baseline scores (logits, not softmax!)
        with torch.no_grad():
            baseline = self.model(x)

        # Handle class_idx properly for batch
        if class_idx is None:
            class_idx = baseline.argmax(dim=1)
        elif isinstance(class_idx, int):
            class_idx = torch.tensor([class_idx] * b, device=device)
        else:
            class_idx = torch.tensor(class_idx, device=device)

        activations = self.activations["value"].to(device)
        _, k, act_h, act_w = activations.size()

        # Upsample all activations
        upsampled_acts = F.interpolate(
            activations, size=(h, w), mode="bilinear", align_corners=False
        )

        # Initialize weights
        weights = torch.zeros(b, k, device=device)

        # Process each activation channel
        for i in range(k):
            act = upsampled_acts[:, i:i+1]
            
            # Skip if activation is constant
            if act.max() == act.min():
                continue
                
            # Normalize activation
            act = (act - act.min()) / (act.max() - act.min() + 1e-8)
            
            score_sum = torch.zeros(b, device=device)
            
            # Accumulate coefficients as per the paper
            coeff = 0.0
            
            for s in range(1, self.num_samples + 1):
                coeff += s / self.num_samples  # Accumulate: sum(j/N) from j=0 to i-1
                
                # Create masked input with accumulated coefficient
                masked_input = x * (coeff * act)
                
                # Process in batches if needed
                for j in range(0, b, self.batch_size):
                    batch_slice = slice(j, min(j + self.batch_size, b))
                    
                    with torch.no_grad():
                        # Get model output (logits, NOT softmax!)
                        masked_output = self.model(masked_input[batch_slice])
                        
                        # Subtract baseline (logit difference)
                        diff = masked_output - baseline[batch_slice]
                        
                        # Get scores for target class
                        if len(class_idx) == 1:
                            score_sum[batch_slice] += diff[:, class_idx[0]]
                        else:
                            score_sum[batch_slice] += torch.gather(
                                diff, 1, class_idx[batch_slice].view(-1, 1)
                            ).squeeze(1)

            # Average over samples
            weights[:, i] = score_sum / self.num_samples

        # Apply softmax to weights
        weights = F.softmax(weights, dim=1)

        # Compute CAM
        cam = torch.zeros((b, 1, h, w), device=device)
        for i in range(k):
            cam += weights[:, i].view(b, 1, 1, 1) * upsampled_acts[:, i:i+1]

        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam_min = cam.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        cam_max = cam.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
