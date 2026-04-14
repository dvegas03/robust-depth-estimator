"""Extensive tests for RobustDepthEstimator — all pipeline stages + Z_est."""

import os

import numpy as np
import pytest

from engine_grounder.geometry.depth_filter import RobustDepthEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _uniform(value, shape=(50, 50)):
    return np.full(shape, value, dtype=np.float32)


def _with_voids(value, shape=(50, 50), void_ratio=0.5, seed=42):
    rng = np.random.default_rng(seed)
    crop = np.full(shape, value, dtype=np.float32)
    n_void = int(crop.size * void_ratio)
    crop.flat[rng.choice(crop.size, n_void, replace=False)] = 0.0
    return crop


def _noisy(value, shape=(100, 100), std=0.1, void_ratio=0.3, seed=0):
    rng = np.random.default_rng(seed)
    d = (np.full(shape, value, dtype=np.float32)
         + rng.normal(0, std, shape).astype(np.float32))
    np.clip(d, 0.0, None, out=d)
    d.flat[rng.choice(d.size, int(d.size * void_ratio), replace=False)] = 0.0
    return d


# ===================================================================
# 1. VOID MASKING
# ===================================================================
class TestVoidMask:
    def test_all_valid(self):
        assert RobustDepthEstimator.void_mask(_uniform(1.0)).all()

    def test_zeros_invalid(self):
        d = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert RobustDepthEstimator.void_mask(d).tolist() == [False, True, False]

    def test_negatives_invalid(self):
        d = np.array([-1.0, 2.0], dtype=np.float32)
        assert RobustDepthEstimator.void_mask(d).tolist() == [False, True]

    def test_inf_invalid(self):
        d = np.array([np.inf, -np.inf, 1.0], dtype=np.float32)
        assert RobustDepthEstimator.void_mask(d).tolist() == [False, False, True]

    def test_nan_invalid(self):
        d = np.array([np.nan, 1.0], dtype=np.float32)
        assert RobustDepthEstimator.void_mask(d).tolist() == [False, True]


# ===================================================================
# 2. ADAPTIVE σ(Z) ESTIMATION
# ===================================================================
class TestEstimateSigmaMap:
    """Validate the depth-binned half-normal sigma estimator."""

    def _make_residuals(self, depth, sigma, seed=0):
        """Simulate residuals ~ |N(0, sigma)| at each valid pixel."""
        rng = np.random.default_rng(seed)
        valid = RobustDepthEstimator.void_mask(depth)
        res = np.zeros_like(depth)
        res[valid] = np.abs(rng.normal(0, sigma, int(valid.sum()))).astype(np.float32)
        return res

    def test_recovers_uniform_sigma(self):
        """Uniform noise σ = 0.02 m should be estimated within ±50%."""
        true_sigma = 0.02
        depth = _uniform(2.5, (80, 80))
        res = self._make_residuals(depth, true_sigma)
        est = RobustDepthEstimator()
        sigma_map, _ = est.estimate_sigma_map(depth, res)
        valid = est.void_mask(depth)
        sigma_est = float(np.median(sigma_map[valid]))
        assert 0.5 * true_sigma <= sigma_est <= 2.0 * true_sigma, (
            f"sigma_est={sigma_est:.4f} far from true {true_sigma}"
        )

    def test_sigma_map_shape(self):
        depth = _uniform(1.5, (30, 40))
        res = self._make_residuals(depth, 0.01)
        est = RobustDepthEstimator()
        sigma_map, thresh_map = est.estimate_sigma_map(depth, res)
        assert sigma_map.shape == depth.shape
        assert thresh_map.shape == depth.shape

    def test_floor_enforced(self):
        """Even with zero residuals the floor must be respected."""
        depth = _uniform(2.0, (20, 20))
        res = np.zeros_like(depth)
        est = RobustDepthEstimator(noise_sigma_floor=0.005)
        sigma_map, _ = est.estimate_sigma_map(depth, res)
        assert (sigma_map[depth > 0] >= 0.005).all()

    def test_thresh_map_equals_k_times_sigma(self):
        depth = _uniform(2.0, (30, 30))
        res = self._make_residuals(depth, 0.02)
        est = RobustDepthEstimator(outlier_k=3.0)
        sigma_map, thresh_map = est.estimate_sigma_map(depth, res)
        valid = depth > 0
        np.testing.assert_array_almost_equal(
            thresh_map[valid], 3.0 * sigma_map[valid], decimal=5
        )

    def test_adapts_to_different_sigma_levels(self):
        """Estimator should produce higher σ for noisier data."""
        depth = _uniform(2.0, (80, 80))
        est = RobustDepthEstimator()
        s1, _ = est.estimate_sigma_map(depth, self._make_residuals(depth, 0.005))
        s2, _ = est.estimate_sigma_map(depth, self._make_residuals(depth, 0.05))
        valid = depth > 0
        assert np.median(s2[valid]) > np.median(s1[valid]), (
            "Higher-noise data should yield higher σ estimate"
        )

    def test_all_void_returns_floor(self):
        depth = np.zeros((20, 20), dtype=np.float32)
        res = np.zeros_like(depth)
        est = RobustDepthEstimator(noise_sigma_floor=0.003)
        sigma_map, _ = est.estimate_sigma_map(depth, res)
        assert (sigma_map >= 0.003).all()


# ===================================================================
# 3. ADAPTIVE BILATERAL OUTLIER DETECTION
# ===================================================================
class TestBilateralOutlier:

    def setup_method(self):
        self.est = RobustDepthEstimator(outlier_k=3.0)

    def test_returns_three_tuple(self):
        result = self.est.bilateral_outlier_mask(_uniform(2.0, (30, 30)))
        assert len(result) == 3

    def test_uniform_no_outliers(self):
        mask, _, _ = self.est.bilateral_outlier_mask(_uniform(2.0, (50, 50)))
        assert mask.sum() == 0

    def test_spike_detected(self):
        d = _uniform(2.0, (30, 30))
        d[15, 15] = 5.0
        mask, _, _ = self.est.bilateral_outlier_mask(d)
        assert mask[15, 15], "Extreme spike should be flagged"

    def test_gradual_change_not_outlier(self):
        d = np.linspace(2.0, 3.0, 50*50, dtype=np.float32).reshape(50, 50)
        mask, _, _ = self.est.bilateral_outlier_mask(d)
        assert mask.sum() / d.size < 0.02

    def test_bilateral_filter_zeros_outliers(self):
        d = _uniform(2.0, (30, 30))
        d[15, 15] = 10.0
        out = self.est.bilateral_filter(d)
        assert out[15, 15] == 0.0
        assert out[0, 0] == pytest.approx(2.0)

    def test_empty_depth(self):
        d = np.zeros((20, 20), dtype=np.float32)
        mask, sm, tm = self.est.bilateral_outlier_mask(d)
        assert mask.sum() == 0

    def test_sigma_map_positive_where_valid(self):
        d = _uniform(2.0, (30, 30))
        _, sigma_map, _ = self.est.bilateral_outlier_mask(d)
        assert (sigma_map[d > 0] > 0).all()

    def test_thresh_map_equals_k_sigma(self):
        d = _uniform(2.0, (40, 40))
        est = RobustDepthEstimator(outlier_k=4.0)
        _, sigma_map, thresh_map = est.bilateral_outlier_mask(d)
        valid = d > 0
        np.testing.assert_array_almost_equal(
            thresh_map[valid], 4.0 * sigma_map[valid], decimal=5
        )

    def test_noise_vs_geometry(self):
        """Realistic noise should flag < 10% of valid pixels."""
        rng = np.random.default_rng(42)
        d = np.linspace(2.0, 3.0, 100*100, dtype=np.float32).reshape(100, 100)
        d += rng.normal(0, 0.015, d.shape).astype(np.float32)
        mask, _, _ = self.est.bilateral_outlier_mask(d)
        n_valid = (d > 0).sum()
        assert mask.sum() / n_valid < 0.10

    def test_no_noise_sigma_parameter_needed(self):
        """Adaptive estimation should work without providing noise_sigma."""
        rng = np.random.default_rng(1)
        d = (_uniform(2.0, (60, 60))
             + rng.normal(0, 0.008, (60, 60)).astype(np.float32))
        np.clip(d, 0.0, None, out=d)
        # No noise_sigma param — should not raise
        est = RobustDepthEstimator(outlier_k=3.0)
        mask, sigma_map, thresh_map = est.bilateral_outlier_mask(d)
        assert sigma_map is not None
        assert (sigma_map[d > 0] > 0).all()


# ===================================================================
# 4. IQR FILTER (legacy utility)
# ===================================================================
class TestIQRFilter:
    def setup_method(self):
        self.est = RobustDepthEstimator()

    def test_uniform_untouched(self):
        d = _uniform(3.0)
        np.testing.assert_array_equal(self.est.iqr_filter(d, 1.5), d)

    def test_outlier_zeroed(self):
        d = _uniform(1.0, (10, 10))
        d[0, 0] = 100.0
        out = self.est.iqr_filter(d, 1.5)
        assert out[0, 0] == 0.0
        assert out[5, 5] == 1.0

    def test_does_not_modify_input(self):
        d = _uniform(1.0, (10, 10))
        d[0, 0] = 100.0
        orig = d.copy()
        self.est.iqr_filter(d, 1.5)
        np.testing.assert_array_equal(d, orig)

    def test_all_void_passthrough(self):
        d = np.zeros((10, 10), dtype=np.float32)
        np.testing.assert_array_equal(self.est.iqr_filter(d, 1.5), d)


# ===================================================================
# 5. MORPHOLOGICAL CLOSING
# ===================================================================
class TestMorphClose:
    def setup_method(self):
        self.est = RobustDepthEstimator(morph_kernel=5)

    def test_no_voids_unchanged(self):
        d = _uniform(2.0)
        np.testing.assert_array_almost_equal(self.est.morph_close(d), d, decimal=5)

    def test_fills_single_pixel_void(self):
        d = _uniform(2.0, (20, 20))
        d[10, 10] = 0.0
        out = self.est.morph_close(d)
        assert out[10, 10] > 0.0
        assert out[10, 10] == pytest.approx(2.0, abs=0.1)

    def test_kernel_zero_disables(self):
        est = RobustDepthEstimator(morph_kernel=0)
        d = _uniform(2.0, (20, 20))
        d[10, 10] = 0.0
        assert est.morph_close(d)[10, 10] == 0.0


# ===================================================================
# 6. INPAINTING
# ===================================================================
class TestInpaint:
    def setup_method(self):
        self.est = RobustDepthEstimator(inpaint_radius=5)

    def test_no_voids_passthrough(self):
        d = _uniform(1.5)
        np.testing.assert_array_almost_equal(self.est.inpaint(d), d, decimal=4)

    def test_fills_all_voids(self):
        d = _with_voids(2.0, (50, 50), 0.3)
        assert (self.est.inpaint(d) > 0).all()

    def test_original_valid_preserved(self):
        d = _with_voids(2.0, (50, 50), 0.3)
        valid = d > 0
        out = self.est.inpaint(d)
        np.testing.assert_array_almost_equal(out[valid], d[valid], decimal=4)

    def test_telea_vs_ns(self):
        d = _with_voids(2.0, (30, 30), 0.3)
        assert (self.est.inpaint(d, "telea") > 0).all()
        assert (self.est.inpaint(d, "ns") > 0).all()


# ===================================================================
# 7. BILATERAL SMOOTHING
# ===================================================================
class TestBilateralSmooth:
    def setup_method(self):
        self.est = RobustDepthEstimator(
            bilateral_d=5, bilateral_sigma_color=0.04, bilateral_sigma_space=4.5
        )

    def test_uniform_unchanged(self):
        d = _uniform(3.0, (50, 50))
        np.testing.assert_array_almost_equal(
            self.est.bilateral_smooth(d), d, decimal=5
        )

    def test_reduces_noise(self):
        rng = np.random.default_rng(0)
        d = (np.full((100, 100), 2.0, dtype=np.float32)
             + rng.normal(0, 0.05, (100, 100)).astype(np.float32))
        assert self.est.bilateral_smooth(d).std() < d.std()

    def test_preserves_edges(self):
        d = np.zeros((50, 50), dtype=np.float32)
        d[:, :25] = 1.0
        d[:, 25:] = 3.0
        out = self.est.bilateral_smooth(d)
        assert out[25, 5]  == pytest.approx(1.0, abs=0.1)
        assert out[25, 45] == pytest.approx(3.0, abs=0.1)


# ===================================================================
# 8. FULL PIPELINE (restore)
# ===================================================================
class TestRestore:
    def setup_method(self):
        self.est = RobustDepthEstimator()

    def test_output_shape(self):
        d = _noisy(2.0)
        assert self.est.restore(d).shape == d.shape

    def test_output_all_positive(self):
        assert (self.est.restore(_noisy(2.0, void_ratio=0.5)) > 0).all()

    def test_closer_to_truth_than_input(self):
        true_val = 2.0
        d = _noisy(true_val, void_ratio=0.4, std=0.08)
        out = self.est.restore(d)
        both = (d > 0) & (out > 0)
        rmse_b = np.sqrt(np.mean((d[both] - true_val)**2))
        rmse_a = np.sqrt(np.mean((out[both] - true_val)**2))
        assert rmse_a <= rmse_b + 0.01


# ===================================================================
# 9. Z_est EXTRACTION (get_stable_z)
# ===================================================================
class TestGetStableZ:
    def setup_method(self):
        self.est = RobustDepthEstimator(tau_threshold=0.10)

    def test_uniform_no_voids(self):
        assert self.est.get_stable_z(_uniform(2.5)) == pytest.approx(2.5)

    def test_uniform_with_voids(self):
        assert self.est.get_stable_z(
            _with_voids(1.0, void_ratio=0.5)
        ) == pytest.approx(1.0)

    def test_returns_float(self):
        assert isinstance(self.est.get_stable_z(_uniform(3.3)), float)

    def test_empty_array(self):
        assert self.est.get_stable_z(np.array([]).reshape(0, 0)) is None

    def test_all_zeros(self):
        assert self.est.get_stable_z(np.zeros((50, 50))) is None

    def test_single_pixel(self):
        assert self.est.get_stable_z(np.array([[7.7]])) == pytest.approx(7.7)

    def test_zeros_excluded(self):
        d = np.array([[0.0, 0.0, 1.0, 1.0, 1.0]])
        assert self.est.get_stable_z(d) == pytest.approx(1.0)

    def test_negatives_excluded(self):
        d = np.array([[-5.0, -1.0, 2.0, 2.0, 2.0]])
        assert self.est.get_stable_z(d) == pytest.approx(2.0)

    def test_inf_excluded(self):
        d = np.array([[np.inf, -np.inf, 0.7, 0.7, 0.7]])
        assert self.est.get_stable_z(d) == pytest.approx(0.7)

    def test_nan_excluded(self):
        d = np.array([[np.nan, np.nan, 0.4, 0.4, 0.4]])
        assert self.est.get_stable_z(d) == pytest.approx(0.4)

    def test_single_outlier_rejected(self):
        d = _uniform(1.0, (30, 30))
        d[15, 15] = 100.0
        assert self.est.get_stable_z(d) == pytest.approx(1.0, abs=0.05)

    def test_exactly_at_tau(self):
        d = np.zeros(100, dtype=np.float32)
        d[:10] = 2.0
        assert self.est.get_stable_z(d.reshape(10, 10)) == pytest.approx(2.0)

    def test_below_tau_no_context(self):
        d = _with_voids(1.0, (100, 100), void_ratio=0.96)
        assert RobustDepthEstimator(tau_threshold=0.10).get_stable_z(d) is None

    def test_fallback_with_full_map(self):
        full = np.full((100, 100), 3.0, dtype=np.float32)
        crop = np.zeros((10, 10), dtype=np.float32)
        z = RobustDepthEstimator(tau_threshold=0.50).get_stable_z(
            crop, full_depth_map=full, bbox=(40, 50, 40, 50)
        )
        assert z == pytest.approx(3.0)


# ===================================================================
# 10. STATISTICAL ACCURACY (Monte Carlo)
# ===================================================================
class TestStatisticalAccuracy:
    def test_noisy_recovery(self):
        true_z = 2.0
        est = RobustDepthEstimator(tau_threshold=0.05, outlier_k=3.0)
        errors = [
            abs(est.get_stable_z(_noisy(true_z, (50, 50), 0.1, 0.4, i)) - true_z)
            for i in range(50)
        ]
        assert np.median(errors) < 0.05

    def test_high_void_recovery(self):
        true_z = 1.5
        est = RobustDepthEstimator(tau_threshold=0.05, outlier_k=3.0)
        errors = [
            abs(est.get_stable_z(_noisy(true_z, (80, 80), 0.05, 0.85, i+100)) - true_z)
            for i in range(30)
            if est.get_stable_z(_noisy(true_z, (80, 80), 0.05, 0.85, i+100)) is not None
        ]
        assert len(errors) > 0
        assert np.median(errors) < 0.10

    def test_adapts_to_unknown_noise_level(self):
        """Adaptive estimation should achieve good Z accuracy without knowing sigma."""
        true_z = 2.5
        results = []
        for sigma in [0.005, 0.015, 0.03]:
            est = RobustDepthEstimator(outlier_k=3.0)   # no noise_sigma
            errs = []
            for i in range(20):
                d = _noisy(true_z, (60, 60), sigma, 0.4, seed=i*7)
                z = est.get_stable_z(d)
                if z is not None:
                    errs.append(abs(z - true_z))
            results.append(np.median(errs) if errs else np.inf)
        # Each noise level should achieve < 5 cm median error
        for sigma, err in zip([0.005, 0.015, 0.03], results):
            assert err < 0.05, f"sigma={sigma}: median error {err:.4f} >= 0.05"
