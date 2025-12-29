from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def visualize_cam(
    img: torch.Tensor,          # (1,3,H,W) – NOT normalized
    cams: dict,                 # {"GradCAM": (1,1,H,W), ...}
    figure_title: str | None = None,
    save_path: str | None = None,
    alpha: float = 0.5,
    nrow: int | None = None     # number of CAMs per row
):
    """
    Display CAM overlays using matplotlib subplots with optional nrow.

    Args:
        img: input image tensor (1,3,H,W), values in [0,1]
        cams: dict of CAM tensors (1,1,H,W)
        figure_title: global figure title
        save_path: path to save figure
        alpha: CAM overlay strength
        nrow: number of CAMs per row (Original image + CAMs)
    """

    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,3)
    n_images = 1 + len(cams)  # original + CAMs

    if nrow is None:
        nrow = n_images  # all in one row

    ncol = min(nrow, n_images)
    nline = math.ceil(n_images / nrow)

    fig, axes = plt.subplots(nline, ncol, figsize=(3 * ncol, 3 * nline))
    axes = np.array(axes).reshape(-1)  # flatten in case of single row/col

    if figure_title:
        fig.suptitle(figure_title, fontsize=16)
        plt.subplots_adjust(top=0.9)


    # ---- Original image ----
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # ---- CAM overlays ----
    for idx, (name, cam) in enumerate(cams.items(), start=1):
        ax = axes[idx]
        cam_np = cam.squeeze().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_np),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        overlay = (1 - alpha) * img_np + alpha * heatmap

        ax.imshow(overlay)
        ax.set_title(name)
        ax.axis("off")

    # Turn off extra axes if any
    for ax in axes[n_images:]:
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()


def convert_to_gray(x, percentile=99):
    """
    Args:
        x: torch tensor with shape of (1, 3, H, W)
        percentile: int
    Return:
        result: shape of (1, 1, H, W)
    """
    x_2d = torch.abs(x).sum(dim=1).squeeze(0)
    v_max = np.percentile(x_2d.cpu().numpy(), percentile)
    v_min = torch.min(x_2d)
    torch.clamp_((x_2d - v_min) / (v_max - v_min), 0, 1)
    return x_2d.unsqueeze(0).unsqueeze(0)
