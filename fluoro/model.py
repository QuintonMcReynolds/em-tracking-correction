from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(), nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())


class UNet(nn.Module):
    """Small residual U-Net. Input: [noisy normalised counts, per-pixel noise-std estimate]. Output: transmission."""

    def __init__(self, c: int = 32):
        super().__init__()
        self.e1, self.e2, self.e3 = _block(2, c), _block(c, 2 * c), _block(2 * c, 4 * c)
        self.pool = nn.MaxPool2d(2)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), _block(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), _block(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return x[:, :1] + self.out(d1)          # residual on the noisy input


def make_input(y: torch.Tensor, n0: torch.Tensor, n_frames: int = 1) -> torch.Tensor:
    """y (B,1,H,W) normalised counts (mean of n_frames), n0 (B,) -> (B,2,H,W) with a Poisson noise-std map."""
    std = torch.sqrt(torch.clamp(y, min=1e-4) / (n0[:, None, None, None] * n_frames))
    return torch.cat([y, std * 10.0], 1)
