# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import convert_to_gray


class IntegratedGradients:
    """
    Integrated Gradients for attributing model predictions to input features.

    Reference:
        Sundararajan et al., "Axiomatic Attribution for Deep Networks"
        ICML 2017. https://arxiv.org/abs/1703.01365
    """

    def __init__(self, model: nn.Module, n_steps: int = 50):
        """
        Args:
            model: Trained neural network model.
            n_steps: Number of steps for path integral approximation.
        """
        self.model = model.eval()
        self.n_steps = n_steps

    def forward(
        self,
        x: torch.Tensor,
        x_baseline: torch.Tensor | None = None,
        class_idx: int | None = None,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients attribution map.

        Args:
            x: Input tensor of shape (B, C, H, W)
            x_baseline: Baseline tensor (same shape as x)
            class_idx: Target class index (None → predicted class)

        Returns:
            Integrated Gradients attribution map
        """
        device = x.device

        if x_baseline is None:
            x_baseline = torch.zeros_like(x)
        else:
            x_baseline = x_baseline.to(device)

        assert x.shape == x_baseline.shape

        # Difference from baseline
        x_diff = x - x_baseline

        # Accumulate gradients
        total_grads = torch.zeros_like(x)

        # α ∈ (0, 1]
        alphas = torch.linspace(
            0.0, 1.0, steps=self.n_steps + 1, device=device
        )[1:]

        for alpha in alphas:
            x_step = (x_baseline + alpha * x_diff).requires_grad_(True)

            output = self.model(x_step)

            if class_idx is None:
                idx = output.argmax(dim=1)
                score = output[torch.arange(output.size(0)), idx]
            else:
                score = output[:, class_idx]

            self.model.zero_grad()
            score.sum().backward()

            if x_step.grad is not None:
                total_grads += x_step.grad

        # Average gradient along path
        avg_grads = total_grads / self.n_steps

        # Integrated gradients
        attributions = x_diff * avg_grads

        # ---- Optional visualization post-processing ----
        attributions = convert_to_gray(attributions.detach().cpu())

        min_val, max_val = attributions.min(), attributions.max()
        if (max_val - min_val) > 0:
            attributions = (attributions - min_val) / (max_val - min_val)

        return attributions

    def __call__(self, x, x_baseline=None, class_idx=None):
        return self.forward(x, x_baseline, class_idx)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
