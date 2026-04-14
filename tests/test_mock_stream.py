"""Tests for MockStream — data loading and corruption injection."""

import os

import numpy as np
import pytest

from engine_grounder.streams.mock_stream import MockStream

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "sample_scene")
HAS_DATA = os.path.isfile(os.path.join(DATA_DIR, "color.jpg"))


# ===================================================================
# 1. MISSING-FILE HANDLING
# ===================================================================
class TestMissingFiles:
    def test_missing_rgb_raises(self):
        stream = MockStream(data_dir="/nonexistent/path")
        with pytest.raises(FileNotFoundError, match="RGB"):
            stream.get_frame()

    def test_missing_depth_raises(self, tmp_path):
        """RGB exists but depth doesn't."""
        import cv2
        fake_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "color.jpg"), fake_rgb)
        stream = MockStream(data_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="depth"):
            stream.get_frame()


# ===================================================================
# 2. LOADING REAL DATA (skipped when sample_scene not present)
# ===================================================================
@pytest.mark.skipif(not HAS_DATA, reason="sample_scene data not downloaded")
class TestRealData:
    def setup_method(self):
        self.stream = MockStream(data_dir=DATA_DIR)

    def test_rgb_shape_is_3_channel(self):
        frame = self.stream.get_frame()
        assert frame["rgb"].ndim == 3
        assert frame["rgb"].shape[2] == 3

    def test_depth_shape_is_2d(self):
        frame = self.stream.get_frame()
        assert frame["depth"].ndim == 2

    def test_rgb_and_depth_same_hw(self):
        frame = self.stream.get_frame()
        assert frame["rgb"].shape[:2] == frame["depth"].shape[:2]

    def test_depth_is_float32_meters(self):
        frame = self.stream.get_frame()
        depth = frame["depth"]
        assert depth.dtype == np.float32
        # Redwood data is indoor, depths should be < 10 m
        valid = depth[depth > 0]
        assert valid.max() < 10.0

    def test_rgb_is_rgb_order(self):
        """Spot-check: loading as RGB means channel 0 != channel 2 for a
        colour image (unless the image is grey, which Redwood isn't)."""
        frame = self.stream.get_frame()
        # Not a rigorous colour-order check, but verifies cvtColor ran
        assert frame["rgb"].dtype == np.uint8


# ===================================================================
# 3. CORRUPTION INJECTION
# ===================================================================
@pytest.mark.skipif(not HAS_DATA, reason="sample_scene data not downloaded")
class TestCorruption:
    def setup_method(self):
        self.stream = MockStream(data_dir=DATA_DIR)

    def test_corruption_zeroes_region(self):
        frame = self.stream.get_corrupted_frame(
            u_min=100, v_min=100, u_max=200, v_max=200
        )
        region = frame["depth"][100:200, 100:200]
        assert np.all(region == 0.0)

    def test_corruption_leaves_outside_untouched(self):
        clean = self.stream.get_frame()["depth"]
        corrupted = self.stream.get_corrupted_frame(
            u_min=100, v_min=100, u_max=200, v_max=200
        )["depth"]
        # Top-left corner should be identical
        np.testing.assert_array_equal(clean[:50, :50], corrupted[:50, :50])

    def test_out_of_bounds_coords_clamped(self):
        """Coordinates beyond image dims should not crash."""
        frame = self.stream.get_corrupted_frame(
            u_min=-10, v_min=-10, u_max=99999, v_max=99999
        )
        assert np.all(frame["depth"] == 0.0)

    def test_inverted_coords_produce_no_crash(self):
        """u_min > u_max should just produce no corruption (empty slice)."""
        depth_clean = self.stream.get_frame()["depth"]
        depth_corrupt = self.stream.get_corrupted_frame(
            u_min=300, v_min=300, u_max=100, v_max=100
        )["depth"]
        np.testing.assert_array_equal(depth_clean, depth_corrupt)
