"""Tests for ShapeEncoder — PointNet mini wrapper."""

import os

import numpy as np
import pytest

from engine_grounder.perception.shape_encoder import ShapeEncoder, PointNetEncoder


def _random_cloud(n=512, seed=42):
    return np.random.default_rng(seed).standard_normal((n, 3)).astype(np.float32)


class TestShapeEncoder:
    def setup_method(self):
        self.enc = ShapeEncoder(embed_dim=256, device="cpu")

    def test_encode_shape(self):
        emb = self.enc.encode(_random_cloud())
        assert emb.shape == (256,)

    def test_encode_dtype(self):
        emb = self.enc.encode(_random_cloud())
        assert emb.dtype == np.float32

    def test_encode_per_point_shape(self):
        n = 512
        pp = self.enc.encode_per_point(_random_cloud(n=n))
        assert pp.shape == (n, 1024)

    def test_deterministic(self):
        pts = _random_cloud()
        e1 = self.enc.encode(pts)
        e2 = self.enc.encode(pts)
        np.testing.assert_array_equal(e1, e2)

    def test_different_inputs_different_outputs(self):
        e1 = self.enc.encode(_random_cloud(seed=0))
        e2 = self.enc.encode(_random_cloud(seed=1))
        assert not np.allclose(e1, e2)

    def test_small_cloud(self):
        emb = self.enc.encode(_random_cloud(n=16))
        assert emb.shape == (256,)

    def test_large_cloud(self):
        emb = self.enc.encode(_random_cloud(n=4096))
        assert emb.shape == (256,)


class TestPointNetEncoder:
    def test_forward_shapes(self):
        import torch
        model = PointNetEncoder(embed_dim=128)
        x = torch.randn(2, 3, 256)
        emb, pp = model(x)
        assert emb.shape == (2, 128)
        assert pp.shape == (2, 1024, 256)

    def test_single_batch(self):
        import torch
        model = PointNetEncoder(embed_dim=64)
        x = torch.randn(1, 3, 100)
        emb, pp = model(x)
        assert emb.shape == (1, 64)
