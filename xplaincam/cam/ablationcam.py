from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from .basecam import BaseCAM

class AblationCAM(BaseCAM):
    
    """
    Ablation-CAM: computes the contribution score of each feature map by evaluating the decrease in the target class score 
    when that feature map is ablated. These contributions are then used to form a weighted combination of activations that 
    highlights class-relevant regions.

    Reference:
        Desai et al., "Ablation-CAM: Visual Explanations for Deep Convolutional Network via Gradient-free Localization."
        https://openaccess.thecvf.com/content_WACV_2020/papers/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.pdf
    """

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        batch_size: int = 32, 
    ) -> torch.Tensor:
        
        device = x.device
        self.model.eval()
        
        if batch_size < 1:
            batch_size = 32

        # 1. Original forward pass to get baseline
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Official code uses the raw score for normalization
        original_score = output[0, class_idx].item()
        
        # Get activations: [1, C, H, W]
        activations = self.activations["value"]
        C = activations.shape[1]
        
        # 2. Compute importance weights via Ablation
        # weights = (Original_Score - Ablated_Score) / Original_Score
        weights = torch.zeros(C, device=device)
        
        for i in range(0, C, batch_size):
            end_i = min(i + batch_size, C)
            current_batch_size = end_i - i
            
            batch_x = x.repeat(current_batch_size, 1, 1, 1)
            
            # This hook replicates the "AblationLayer" behavior
            def ablation_hook(module, input, output):
                # Copy output to avoid modifying the original activation tensor in-place 
                # if it's cached elsewhere
                ablated_output = output.clone()
                for j in range(ablated_output.shape[0]):
                    # Zero out the specific channel for this batch element
                    ablated_output[j, i + j, :, :] = 0
                return ablated_output

            layer = self._get_module_by_name(self.target_layer_name)
            handle = layer.register_forward_hook(ablation_hook)
            
            batch_output = self.model(batch_x)
            handle.remove()
            
            batch_scores = batch_output[:, class_idx]
            
            # MATCH OFFICIAL FORMULA: (original - ablated) / original
            # Note: The official implementation does NOT use ReLU on weights here
            weights[i:end_i] = (original_score - batch_scores) / (original_score + 1e-8)

        # 3. Generate Heatmap
        # Reshape weights to [1, C, 1, 1] for broadcasting
        weights = weights.view(1, C, 1, 1)
        
        # Weighted combination: Sum(w_i * A_i)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU to the final combined heatmap (Standard CAM practice)
        cam = F.relu(cam)

        # 4. Upsample and Normalize
        cam = F.interpolate(
            cam, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        
        # Min-Max Normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        return cam