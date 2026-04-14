"""
Minimal PointNet encoder for 3-D point clouds.

Architecture:
    Input (N, 3) -> SharedMLP [3->64->128->1024] -> MaxPool -> FC [1024->512->256]

Produces:
    - A 256-d global shape embedding.
    - (Optional) N x 1024 per-point feature map for visualisation.

Weights are randomly initialised by default.  A pretrained ModelNet40 checkpoint
can be loaded via ``load_state_dict`` if available.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _PointNetBackbone(nn.Module):
    """Shared-MLP backbone that outputs per-point and global features."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 3, N)

        Returns
        -------
        per_point : (B, 1024, N) — per-point features before pooling.
        global_feat : (B, 1024)  — max-pooled global feature.
        """
        per_point = self.mlp(x)
        global_feat = per_point.max(dim=2).values
        return per_point, global_feat


class PointNetEncoder(nn.Module):
    """Full encoder: backbone + FC head -> 256-d embedding."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.backbone = _PointNetBackbone()
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 3, N)

        Returns
        -------
        embedding  : (B, embed_dim)
        per_point  : (B, 1024, N)
        """
        per_point, glob = self.backbone(x)
        embedding = self.head(glob)
        return embedding, per_point


class ShapeEncoder:
    """
    Numpy-friendly wrapper around :class:`PointNetEncoder`.

    Usage::

        enc = ShapeEncoder()
        emb = enc.encode(points)           # (256,)
        pp  = enc.encode_per_point(points)  # (N, 1024)
    """

    def __init__(self, embed_dim: int = 256, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = PointNetEncoder(embed_dim=embed_dim).to(self.device).eval()

    @torch.no_grad()
    def _forward(self, points: np.ndarray):
        pts = torch.from_numpy(points.astype(np.float32))
        if pts.ndim == 2:
            pts = pts.unsqueeze(0)                 # (1, N, 3)
        x = pts.permute(0, 2, 1).to(self.device)  # (B, 3, N)
        emb, pp = self.model(x)
        return emb, pp

    def encode(self, points: np.ndarray) -> np.ndarray:
        """(N, 3) -> (embed_dim,) global embedding."""
        emb, _ = self._forward(points)
        return emb.squeeze(0).cpu().numpy()

    def encode_per_point(self, points: np.ndarray) -> np.ndarray:
        """(N, 3) -> (N, 1024) per-point features."""
        _, pp = self._forward(points)
        return pp.squeeze(0).permute(1, 0).cpu().numpy()
