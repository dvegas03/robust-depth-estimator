"""
Edge-case tests: empty arrays, all-NaN, sub-kernel images, coplanar points,
degenerate meshes. Ensures the pipeline degrades gracefully on Apple Silicon
instead of triggering Open3D C++ segfaults.
"""

import os

import math
import numpy as np
import open3d as o3d
import pytest

from engine_grounder.geometry.depth_filter import RobustDepthEstimator
from engine_grounder.geometry.mesh_builder import MeshBuilder
from engine_grounder.perception.shape_descriptor import ShapeDescriptor
from engine_grounder.perception.shape_encoder import ShapeEncoder


# ══════════════════════════════════════════════════════════════════════════════
# 1. Depth Filter Edge Cases
# ══════════════════════════════════════════════════════════════════════════════
class TestDepthFilterEdgeCases:
    @pytest.fixture
    def estimator(self):
        return RobustDepthEstimator(bilateral_d=9)

    def test_all_zeros_depth(self, estimator):
        """Test how the filter handles an image with absolutely no valid depth."""
        depth = np.zeros((100, 100), dtype=np.float32)
        outlier_mask, sigma, thresh = estimator.bilateral_outlier_mask(depth)

        assert not outlier_mask.any(), "Zeros should not be flagged as outliers"
        assert sigma.max() == 0.0, "Sigma should be 0 for empty image"
        assert estimator.get_stable_z(depth) is None, "Stable Z should be None"

    def test_all_nan_inf_depth(self, estimator):
        """Test resilience against corrupted float buffers."""
        depth = np.full((50, 50), np.nan, dtype=np.float32)
        depth[10:20, 10:20] = np.inf

        mask = estimator.void_mask(depth)
        assert not mask.any(), "NaN and Inf must be masked out"

        outlier_mask, _, _ = estimator.bilateral_outlier_mask(depth)
        assert not outlier_mask.any(), "Should not crash on NaN/Inf math"

    def test_micro_depth_map(self, estimator):
        """Test array smaller than the bilateral filter kernel size (d=9)."""
        depth = np.ones((5, 5), dtype=np.float32)

        # Should gracefully skip bilateral processing and return the raw median
        z_est = estimator.get_stable_z(depth)
        assert z_est == 1.0, "Micro arrays should fallback to raw median"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Mesh Builder Edge Cases (The Crash Zone)
# ══════════════════════════════════════════════════════════════════════════════
class TestMeshBuilderEdgeCases:
    @pytest.fixture
    def builder(self):
        return MeshBuilder(voxel_size=0.01)

    def test_empty_point_cloud(self, builder):
        """Test absolutely empty geometry."""
        pts = np.empty((0, 3), dtype=np.float64)
        mesh = builder.from_point_cloud(pts)
        assert len(mesh.vertices) == 0, "Must return empty mesh safely"

    def test_under_30_points(self, builder):
        """Test the exact boundary condition that crashed Open3D's KNN."""
        # 29 points -> triggers the guard before normal estimation
        pts = np.random.rand(29, 3)
        mesh = builder.from_point_cloud(pts)
        assert len(mesh.vertices) == 0, "Must abort if under 30 points"

    def test_collapse_during_voxelization(self, builder):
        """Test when points exist, but voxel downsampling destroys them."""
        # 100 points, but they are all at the exact same coordinate.
        # Voxel downsampling will merge them into exactly 1 point.
        pts = np.ones((100, 3), dtype=np.float64)
        mesh = builder.from_point_cloud(pts)
        assert len(mesh.vertices) == 0, "Must abort if voxelization drops count < 30"

    def test_simplify_empty_mesh(self, builder):
        """Ensure decimation doesn't crash on 0 faces."""
        mesh = o3d.geometry.TriangleMesh()
        simp = builder.simplify(mesh, target_faces=100)
        assert len(simp.triangles) == 0

    def test_sample_empty_mesh(self, builder):
        """Ensure surface sampling doesn't crash on 0 faces."""
        mesh = o3d.geometry.TriangleMesh()
        pts = builder.sample_points(mesh, n=2048)
        assert pts.shape == (2048, 3)
        assert np.all(pts == 0), "Should return zeroed array for empty mesh"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Shape Descriptor Edge Cases (The Qhull Trap)
# ══════════════════════════════════════════════════════════════════════════════
class TestShapeDescriptorEdgeCases:
    def test_empty_mesh_description(self):
        """Test description of a completely empty mesh."""
        mesh = o3d.geometry.TriangleMesh()
        desc = ShapeDescriptor.describe(mesh)

        assert math.isnan(desc["volume"])
        assert desc["surface_area"] == 0.0
        assert desc["n_vertices"] == 0
        assert ShapeDescriptor.classify(desc) == "unknown (non-watertight)"

    def test_flat_coplanar_mesh(self):
        """Test 2D plane in 3D space (This traditionally crashes Qhull)."""
        mesh = o3d.geometry.TriangleMesh()
        # A perfectly flat square on the Z=0 plane
        mesh.vertices = o3d.utility.Vector3dVector(
            np.array(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                dtype=np.float64,
            )
        )
        mesh.triangles = o3d.utility.Vector3iVector(
            np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        )

        desc = ShapeDescriptor.describe(mesh)

        # Volume of a 2D plane is 0 or NaN, but it shouldn't segfault
        assert math.isnan(desc["volume"]) or desc["volume"] == 0.0
        assert desc["n_vertices"] == 4


# ══════════════════════════════════════════════════════════════════════════════
# 4. Shape Encoder Edge Cases
# ══════════════════════════════════════════════════════════════════════════════
class TestShapeEncoderEdgeCases:
    @pytest.fixture
    def encoder(self):
        return ShapeEncoder(device="cpu")  # Force CPU to avoid CUDA init overhead in tests

    def test_zero_scale_points(self, encoder):
        """Test PointNet on points that have 0 variance (all identical)."""
        pts = np.zeros((2048, 3), dtype=np.float32)
        emb = encoder.encode(pts)

        assert emb.shape == (256,)
        assert not np.isnan(emb).any(), "Network should not produce NaNs on zero inputs"

    def test_extreme_scale_points(self, encoder):
        """Test points with massive floating point values."""
        pts = np.random.rand(1024, 3).astype(np.float32) * 1e9
        emb = encoder.encode(pts)

        assert emb.shape == (256,)
        assert not np.isnan(emb).any()
