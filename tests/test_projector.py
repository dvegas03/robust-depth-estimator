"""Extensive tests for the Projector (pinhole back-projection math)."""

import os

import numpy as np
import pytest

from engine_grounder.spatial.projector import Projector


# Typical intrinsics for a 640x480 RGB-D sensor
FX, FY = 525.0, 525.0
CX, CY = 319.5, 239.5


# ===================================================================
# 1. CONSTRUCTION VALIDATION
# ===================================================================
class TestConstruction:
    def test_valid_intrinsics(self):
        p = Projector(fx=FX, fy=FY, cx=CX, cy=CY)
        assert p.fx == FX

    def test_zero_fx_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            Projector(fx=0.0, fy=FY, cx=CX, cy=CY)

    def test_zero_fy_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            Projector(fx=FX, fy=0.0, cx=CX, cy=CY)


# ===================================================================
# 2. SINGLE-PIXEL BACK-PROJECTION
# ===================================================================
class TestBackproject:
    def setup_method(self):
        self.proj = Projector(fx=FX, fy=FY, cx=CX, cy=CY)

    def test_center_pixel_projects_to_optical_axis(self):
        """The principal point at any depth should map to X=0, Y=0."""
        pt = self.proj.backproject(int(CX), int(CY), 5.0)
        assert pt[0] == pytest.approx(0.0, abs=0.01)
        assert pt[1] == pytest.approx(0.0, abs=0.01)
        assert pt[2] == pytest.approx(5.0)

    def test_zero_depth_collapses_to_origin(self):
        pt = self.proj.backproject(100, 200, 0.0)
        np.testing.assert_array_almost_equal(pt, [0.0, 0.0, 0.0])

    def test_known_pixel(self):
        """Hand-computed: u=FX+CX, v=CY => X=depth, Y=0."""
        depth = 3.0
        u = int(FX + CX)  # 525 + 319.5 ≈ 844 (but > 640, that's fine for math)
        pt = self.proj.backproject(u, int(CY), depth)
        expected_x = (u - CX) * depth / FX
        assert pt[0] == pytest.approx(expected_x)
        assert pt[1] == pytest.approx(0.0, abs=0.01)

    def test_returns_3d_vector(self):
        pt = self.proj.backproject(0, 0, 1.0)
        assert pt.shape == (3,)

    def test_negative_depth_propagates(self):
        """Projector does pure math; it doesn't reject negative depths."""
        pt = self.proj.backproject(int(CX), int(CY), -2.0)
        assert pt[2] == pytest.approx(-2.0)

    def test_symmetry(self):
        """Pixels equidistant from cx should have opposite X at same depth."""
        d = 4.0
        offset = 50
        pt_left = self.proj.backproject(int(CX) - offset, int(CY), d)
        pt_right = self.proj.backproject(int(CX) + offset, int(CY), d)
        assert pt_left[0] == pytest.approx(-pt_right[0], abs=0.01)


# ===================================================================
# 3. DEPTH-MAP BACK-PROJECTION
# ===================================================================
class TestBackprojectDepthMap:
    def setup_method(self):
        self.proj = Projector(fx=FX, fy=FY, cx=CX, cy=CY)

    def test_output_shape(self):
        depth = np.ones((480, 640), dtype=np.float32)
        cloud = self.proj.backproject_depth_map(depth)
        assert cloud.shape == (480, 640, 3)

    def test_uniform_depth_center_is_zero(self):
        """At the principal point, X and Y should be ~0."""
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        cloud = self.proj.backproject_depth_map(depth)
        cy_i, cx_i = int(CY), int(CX)
        assert cloud[cy_i, cx_i, 0] == pytest.approx(0.0, abs=0.01)
        assert cloud[cy_i, cx_i, 1] == pytest.approx(0.0, abs=0.01)
        assert cloud[cy_i, cx_i, 2] == pytest.approx(2.0)

    def test_all_zero_depth_gives_all_zero_cloud(self):
        depth = np.zeros((100, 100), dtype=np.float32)
        cloud = self.proj.backproject_depth_map(depth)
        np.testing.assert_array_almost_equal(cloud, 0.0)

    def test_single_pixel_matches_backproject(self):
        """The vectorized function must agree with the scalar one."""
        depth = np.full((480, 640), 3.5, dtype=np.float32)
        cloud = self.proj.backproject_depth_map(depth)
        u, v = 200, 150
        scalar = self.proj.backproject(u, v, 3.5)
        np.testing.assert_array_almost_equal(cloud[v, u], scalar)

    def test_rectangular_map(self):
        depth = np.ones((100, 200), dtype=np.float32)
        cloud = self.proj.backproject_depth_map(depth)
        assert cloud.shape == (100, 200, 3)

    def test_z_channel_equals_depth(self):
        """The third channel of the cloud should be the depth map itself."""
        rng = np.random.default_rng(0)
        depth = rng.uniform(0.5, 5.0, (50, 50)).astype(np.float32)
        cloud = self.proj.backproject_depth_map(depth)
        np.testing.assert_array_almost_equal(cloud[:, :, 2], depth, decimal=5)
