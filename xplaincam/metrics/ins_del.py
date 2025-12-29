# # CUDA_VISIBLE_DEVICES=0
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from kornia.filters import gaussian_blur2d
import cv2

class InsertionDeletion:
    """
    Insertion / Deletion evaluation for saliency maps.

    This class separates:
    - metric computation
    - curve storage
    - visualization
    """

    def __init__(
        self,
        model,
        mode="Insertion",
        step=224*2,
        blur_fn=None,
        device=None
    ):
        assert mode in ["Insertion", "Deletion"]
        self.model = model.eval()
        self.mode = mode
        self.step = step
        self.device = device or next(model.parameters()).device
        self.blur_fn = blur_fn or (
            lambda x: gaussian_blur2d(x, (51, 51), (50., 50.))
        )

        self.scores_ = None
        self.auc_ = None

    # --------------------------------------------------
    # Core computation (NO visualization)
    # --------------------------------------------------
    @torch.no_grad()
    def compute(self, image, heatmap, class_idx=None, return_steps=False):
        """
        Compute insertion/deletion scores.

        Args:
            image: (1,C,H,W)
            heatmap: (1,1,H,W) or (H,W)
            return_steps: store full curve

        Returns:
            dict with AUC (+ steps if requested)
        """
        image = image.to(self.device)
        heatmap = heatmap.squeeze()
        H, W = heatmap.shape
        n_pixels = H * W

        if class_idx is None:
            class_idx = self.model(image).argmax(dim=1).item()

        # Initialize images
        if self.mode == "Insertion":
            start = self.blur_fn(image)
            finish = image.clone()
        else:
            start = image.clone()
            finish = torch.zeros_like(image)

        order = torch.argsort(heatmap.view(-1), descending=True)
        n_steps = (n_pixels + self.step - 1) // self.step

        scores = np.zeros(n_steps + 1, dtype=np.float32)

        for i in range(n_steps + 1):
            logits = self.model(start)
            prob = F.softmax(logits, dim=1)[0, class_idx]
            scores[i] = prob.item()

            if i < n_steps:
                idx = order[self.step*i : self.step*(i+1)]
                start.view(1, image.shape[1], -1)[:, :, idx] = \
                    finish.view(1, image.shape[1], -1)[:, :, idx]

        self.scores_ = scores
        self.auc_ = self._auc(scores)

        result = {f"{self.mode}_auc": self.auc_}
        if return_steps:
            result[f"{self.mode}_steps"] = scores

        return result

    # --------------------------------------------------
    # Visualization (optional, explicit)
    # --------------------------------------------------
    @staticmethod
    def plot_curves(results, mode="Insertion", image=None, cam=None, alpha=0.5, title=None, save_path=None):
        """
        Plot insertion/deletion curves.
        - Single CAM  → overlay + curve
        - Multiple CAMs → curves only

        Args:
            results: dict
                - single CAM result
                - OR dict of {name: result}
            mode: "Insertion" or "Deletion"
            image: (1,3,H,W) tensor (required only for single CAM overlay)
            cam: (1,1,H,W) CAM tensor (required only for single CAM overlay)
            alpha: overlay transparency
            title: optional plot title
            save_path: optional path to save the figure
        """

        # Detect single vs multiple CAMs
        is_single = isinstance(results, dict) and f"{mode}_steps" in results

        if is_single:
            if image is None or cam is None:
                raise ValueError("image and cam must be provided for single-CAM plotting")

            steps = results[f"{mode}_steps"]
            auc_val = results[f"{mode}_auc"]
            n = len(steps)
            x = np.linspace(0, 1, n)

            plt.figure(figsize=(10, 5))

            # ---------- LEFT: overlay ----------
            plt.subplot(121)
            img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
            cam_np = cam.squeeze().cpu().numpy()
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            overlay = (1-alpha) * img_np + alpha * heatmap
            plt.imshow(overlay)
            plt.axis("off")
            plt.title(title or "CAM Overlay")

            # ---------- RIGHT: curve ----------
            plt.subplot(122)
            plt.plot(x, steps, label=f"AUC={auc_val:.4f}")
            plt.fill_between(x, 0, steps, alpha=0.4)
            plt.xlabel("Fraction of Pixels")
            plt.ylabel("Model Confidence")
            plt.title(f"{mode} Curve")
            plt.legend()
            plt.grid(True)

        else:
            # Multiple CAMs → curves only
            plt.figure(figsize=(6, 5))
            for name, res in results.items():
                steps = res[f"{mode}_steps"]
                auc_val = res[f"{mode}_auc"]
                x = np.linspace(0, 1, len(steps))
                plt.plot(x, steps, label=f"{name} (AUC={auc_val:.4f})")
            plt.xlabel("Fraction of Pixels")
            plt.ylabel("Model Confidence")
            plt.title(title or f"{mode} Curves Comparison")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        # Always show
        plt.show()
        plt.close()

    # --------------------------------------------------
    # Utils
    # --------------------------------------------------
    @staticmethod
    def _auc(scores):
        return (scores.sum() - scores[0]/2 - scores[-1]/2) / (len(scores) - 1)
