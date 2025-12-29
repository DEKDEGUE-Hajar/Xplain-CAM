from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm

class RISE(nn.Module):
    
    """
    RISE: Generates a set of random binary masks, applies them to the input, 
    feeds masked inputs through the model, and weights the masks by the output score 
    to compute pixel-wise importance.
        
    Reference:
        Petsiuk et al., "RISE: Randomized Input Sampling for Explanation of Black-box Models",
        https://arxiv.org/pdf/1806.07421
    
    """

    def __init__(self, model, input_shape=(3, 224, 224), batch_size=100, N=8000, s=7, p1=0.1, device=None):
        super(RISE, self).__init__()
        assert N % batch_size == 0

        self.model = model.eval()
        self.input_shape = input_shape  # (C, H, W)
        self.batch_size = batch_size
        self.N = N
        self.s = s
        self.p1 = p1
        self.masks = None  # no masks initially

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

    def generate_masks(self):
        """Generate N random masks for RISE."""
        C, H, W = self.input_shape
        cell_size = np.ceil(np.array([H, W]) / self.s)
        up_size = (self.s + 1) * cell_size

        grid = np.random.rand(self.N, self.s, self.s) < self.p1
        grid = grid.astype('float32')

        masks = np.empty((self.N, 1, H, W), dtype=np.float32)

        for i in tqdm(range(self.N), desc="Generating masks"):
            x_shift = np.random.randint(0, int(cell_size[0]))
            y_shift = np.random.randint(0, int(cell_size[1]))
            mask = resize(
                grid[i],
                (int(up_size[0]), int(up_size[1])),
                order=1, mode='reflect', anti_aliasing=False
            )[x_shift:x_shift+H, y_shift:y_shift+W]
            masks[i, 0] = mask

        self.masks = torch.from_numpy(masks).float().to(self.device)

    def forward(self, x, class_idx=None):
        if self.masks is None:
            self.generate_masks()
        N = self.N
        _, _, H, W = x.size()
        x = x.to(self.device)
        saliency = torch.zeros(1, 1, H, W, device=self.device)

        if class_idx is None:
            logit = self.model(x)
            class_idx = logit.argmax(1)
        else:
            class_idx = torch.LongTensor([class_idx]).to(self.device)

        for i in range(0, N, self.batch_size):
            mask = self.masks[i: i + self.batch_size]
            mask = mask.to(self.device)
            with torch.no_grad():
                logit = self.model(mask * x)
            score = logit[:, class_idx].unsqueeze(-1).unsqueeze(-1)
            saliency += (score * mask).sum(dim=0, keepdims=True)

        return saliency / N / self.p1

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass