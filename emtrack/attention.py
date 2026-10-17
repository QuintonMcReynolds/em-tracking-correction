"""Encoder-decoder attention model for few-shot site adaptation.

Encoder: a transformer over the site's K calibration tokens, each = (sensor features, measured correction).
Decoder: every query frame cross-attends to the encoded calibration set (plus learned 'prior' tokens, so K = 0
degrades gracefully to a global model) and regresses the 6-D correction.
No positional encodings: the calibration set is a *set*, so the output must not depend on its order (checked as an
ONNX-package invariant).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossBlock(nn.Module):
    """Pre-norm cross-attention + FFN. Queries attend to the memory only, never to each other, so each frame's
    correction depends only on that frame and the site's calibration set."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.n2, self.nm = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, mem, pad):
        m = self.nm(mem)
        q = q + self.attn(self.n1(q), m, m, key_padding_mask=pad, need_weights=False)[0]
        return q + self.ff(self.n2(q))


class SiteAttention(nn.Module):
    def __init__(self, d_feat: int, d_out: int = 6, d: int = 128, heads: int = 4, enc_layers: int = 3,
                 dec_layers: int = 2, n_prior: int = 4):
        super().__init__()
        self.ctx_in = nn.Sequential(nn.Linear(d_feat + d_out, d), nn.GELU(), nn.Linear(d, d))
        self.q_in = nn.Sequential(nn.Linear(d_feat, d), nn.GELU(), nn.Linear(d, d))
        self.prior = nn.Parameter(torch.randn(1, n_prior, d) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True), enc_layers,
            enable_nested_tensor=False)
        self.decoder = nn.ModuleList([CrossBlock(d, heads) for _ in range(dec_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_out))

    def forward(self, ctx_x, ctx_y, ctx_mask, q_x):
        """ctx_x (B,K,F), ctx_y (B,K,6), ctx_mask (B,K) True = padding, q_x (B,Q,F) -> (B,Q,6). Inputs standardised."""
        B = q_x.shape[0]
        tok = self.ctx_in(torch.cat([ctx_x, ctx_y], -1))
        tok = torch.cat([self.prior.expand(B, -1, -1), tok], 1)
        pad = torch.cat([torch.zeros(B, self.prior.shape[1], dtype=torch.bool, device=q_x.device), ctx_mask], 1)
        mem = self.encoder(tok, src_key_padding_mask=pad)
        h = self.q_in(q_x)
        for blk in self.decoder:
            h = blk(h, mem, pad)
        return self.head(h)
