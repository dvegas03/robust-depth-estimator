"""
Metric-threshold tests for the Engine Grounder pipeline.

Unlike the functional tests in test_depth_filter.py (which test correctness),
these tests assert *quantitative performance bounds* — they fail if the
pipeline regresses below the expected quality bars.

Test groups
-----------
  1. Z Estimation Accuracy          — error < threshold across corruption levels
  2. RMSE Improvement               — filter must improve depth RMSE
  3. Noise Model (σ̂) Accuracy      — adaptive estimator within 50% of truth
  4. Outlier Detection P/R/F1       — controlled spike injection
  5. Inlier Retention               — clean pixels not falsely discarded
  6. Chamfer / Geometry Quality     — point cloud fidelity (requires Open3D)
  7. Shape Classification           — known primitives get correct labels
  8. Full Pipeline Metrics          — end-to-end on flat synthetic scene
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest


from engine_grounder.geometry.depth_filter import RobustDepthEstimator
from benchmarks.metrics import (
    z_error_cm, depth_rmse, depth_mae,
    rmse_improvement_ratio,
    outlier_metrics, inlier_retention_rate,
    sigma_recovery_rel_error,
    chamfer_hausdorff, point_cloud_coverage,
)

# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures / helpers
# ══════════════════════════════════════════════════════════════════════════════

TRUE_DEPTH    = 2.0     # metres — flat synthetic scene
GRID_SHAPE    = (100, 100)
N_SPIKE_PXLS  = 50      # injected outliers for P/R/F1 tests
SPIKE_AMP     = 0.6     # metres


def _flat(void_ratio: float, noise_std: float, seed: int = 0):
    """Return (ground_truth, corrupted, void_mask) for a flat synthetic scene."""
    rng   = np.random.default_rng(seed)
    gt    = np.full(GRID_SHAPE, TRUE_DEPTH, dtype=np.float32)
    noisy = gt + rng.normal(0, noise_std, GRID_SHAPE).astype(np.float32)
    np.clip(noisy, 0, None, out=noisy)

    n_void   = int(noisy.size * void_ratio)
    void_idx = rng.choice(noisy.size, n_void, replace=False)
    noisy.flat[void_idx] = 0.0
    void_mask = np.zeros(GRID_SHAPE, dtype=bool)
    void_mask.flat[void_idx] = True
    return gt, noisy, void_mask


def _with_spikes(void_ratio: float, noise_std: float, seed: int = 0):
    """Return (corrupted, valid_mask, true_spike_mask) after spike injection."""
    rng = np.random.default_rng(seed + 999)
    gt, corrupted, void_mask = _flat(void_ratio, noise_std, seed)
    valid_mask = corrupted > 0
    valid_idx  = np.flatnonzero(valid_mask)
    n_spk      = min(N_SPIKE_PXLS, len(valid_idx))
    spike_idx  = rng.choice(valid_idx, n_spk, replace=False)
    corrupted.flat[spike_idx] += SPIKE_AMP
    spike_mask = np.zeros(GRID_SHAPE, dtype=bool)
    spike_mask.flat[spike_idx] = True
    return corrupted, valid_mask, spike_mask


# ══════════════════════════════════════════════════════════════════════════════
# 1. Z Estimation Accuracy
# ══════════════════════════════════════════════════════════════════════════════

class TestZEstimationAccuracy:
    """Z_est must stay within tolerance across a range of corruption levels."""

    @pytest.fixture(autouse=True)
    def _est(self):
        self.est = RobustDepthEstimator(outlier_k=3.0, tau_threshold=0.10)

    @pytest.mark.parametrize("void_ratio,noise_std,max_err_cm", [
        (0.10, 0.005, 1.0),
        (0.30, 0.015, 2.0),
        (0.50, 0.015, 3.0),
        (0.50, 0.030, 5.0),
        (0.70, 0.015, 5.0),
    ])
    def test_z_error_within_budget(self, void_ratio, noise_std, max_err_cm):
        """Median Z error over 10 seeds must stay below max_err_cm."""
        errors = []
        for seed in range(10):
            _, corrupted, _ = _flat(void_ratio, noise_std, seed)
            z_est = self.est.get_stable_z(corrupted)
            if z_est is not None:
                errors.append(z_error_cm(z_est, TRUE_DEPTH))

        assert len(errors) > 0, "get_stable_z returned None for all seeds"
        med_err = float(np.median(errors))
        assert med_err <= max_err_cm, (
            f"void={void_ratio:.0%}, σ={noise_std:.3f}: "
            f"median Z error {med_err:.3f} cm > limit {max_err_cm} cm"
        )

    def test_z_error_high_void_fallback(self):
        """At 85% voids the estimator may fall back but must still be < 8 cm."""
        errors = []
        for seed in range(10):
            _, corrupted, _ = _flat(0.85, 0.010, seed)
            z_est = self.est.get_stable_z(corrupted)
            if z_est is not None:
                errors.append(z_error_cm(z_est, TRUE_DEPTH))
        if errors:
            assert np.median(errors) < 8.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. RMSE Improvement
# ══════════════════════════════════════════════════════════════════════════════

class TestRMSEImprovement:
    """Filtering must reduce RMSE compared to the raw corrupted map."""

    @pytest.fixture(autouse=True)
    def _est(self):
        self.est = RobustDepthEstimator(outlier_k=3.0)

    @pytest.mark.parametrize("void_ratio,noise_std,min_ratio", [
        (0.30, 0.015, 1.3),
        (0.50, 0.030, 1.2),
        (0.50, 0.050, 1.1),
    ])
    def test_rmse_improves(self, void_ratio, noise_std, min_ratio):
        """
        Inject known spikes, then check that filtering reduces RMSE.

        The bilateral filter is designed to remove outliers (spikes), not to
        reduce Gaussian noise directly — so we must inject explicit spikes to
        produce a measurable RMSE improvement.
        """
        ratios = []
        rng = np.random.default_rng(0)
        for seed in range(10):
            gt, corrupted, _ = _flat(void_ratio, noise_std, seed)
            valid_idx = np.flatnonzero(corrupted > 0)
            n_spk = max(20, int(len(valid_idx) * 0.05))
            spike_idx = rng.choice(valid_idx, min(n_spk, len(valid_idx)), replace=False)
            corrupted.flat[spike_idx] += SPIKE_AMP      # inject large spikes
            valid = corrupted > 0

            omask, _, _ = self.est.bilateral_outlier_mask(corrupted)
            clean        = corrupted.copy()
            clean[omask] = 0.0

            rmse_b = depth_rmse(corrupted, gt, mask=valid)
            rmse_a = depth_rmse(clean,     gt, mask=(clean > 0))
            if not (np.isnan(rmse_b) or np.isnan(rmse_a)):
                ratios.append(rmse_improvement_ratio(rmse_b, rmse_a))

        assert len(ratios) > 0
        assert np.median(ratios) >= min_ratio, (
            f"Median RMSE ratio {np.median(ratios):.3f} < {min_ratio} "
            f"(void={void_ratio:.0%}, σ={noise_std:.3f})"
        )

    def test_mae_positive_after_filter(self):
        """MAE on filtered output must be a positive finite number."""
        gt, corrupted, _ = _flat(0.40, 0.020, seed=0)
        omask, _, _  = self.est.bilateral_outlier_mask(corrupted)
        clean        = corrupted.copy()
        clean[omask] = 0.0
        mae = depth_mae(clean, gt)
        assert np.isfinite(mae) and mae > 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. Noise Model (σ̂) Accuracy
# ══════════════════════════════════════════════════════════════════════════════

class TestSigmaRecovery:
    """The adaptive σ̂(Z) estimator must track the true injected noise."""

    @pytest.fixture(autouse=True)
    def _est(self):
        self.est = RobustDepthEstimator(outlier_k=3.0)

    @pytest.mark.parametrize("noise_std", [0.005, 0.015, 0.030, 0.050])
    def test_sigma_within_50pct(self, noise_std):
        """Median relative error of σ̂ must be < 50% across 10 seeds."""
        rel_errs = []
        for seed in range(10):
            _, corrupted, _ = _flat(0.30, noise_std, seed)
            valid = corrupted > 0
            _, sigma_map, _ = self.est.bilateral_outlier_mask(corrupted)
            if sigma_map is not None and valid.any():
                sigma_hat = float(np.median(sigma_map[valid]))
                rel_errs.append(sigma_recovery_rel_error(sigma_hat, noise_std))

        assert len(rel_errs) > 0
        assert np.median(rel_errs) < 0.50, (
            f"σ={noise_std:.3f}: median rel. error {np.median(rel_errs):.3f} ≥ 0.50"
        )

    def test_sigma_monotone_with_noise(self):
        """A noisier depth map should produce a higher σ̂ estimate."""
        _, d_low,  _ = _flat(0.30, 0.005, seed=0)
        _, d_high, _ = _flat(0.30, 0.050, seed=0)
        _, sm_low,  _ = self.est.bilateral_outlier_mask(d_low)
        _, sm_high, _ = self.est.bilateral_outlier_mask(d_high)
        valid_low  = d_low  > 0
        valid_high = d_high > 0
        assert np.median(sm_high[valid_high]) > np.median(sm_low[valid_low]), (
            "Higher-noise input should yield a higher σ̂"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Outlier Detection Precision / Recall / F1
# ══════════════════════════════════════════════════════════════════════════════

class TestOutlierDetectionMetrics:
    """Controlled spike injection tests for P/R/F1."""

    @pytest.fixture(autouse=True)
    def _est(self):
        self.est = RobustDepthEstimator(outlier_k=3.0)

    def _run(self, void_ratio, noise_std, n_seeds=8):
        p_list, r_list, f1_list = [], [], []
        for seed in range(n_seeds):
            corrupted, valid_mask, spike_mask = _with_spikes(void_ratio, noise_std, seed)
            omask, _, _ = self.est.bilateral_outlier_mask(corrupted)
            m = outlier_metrics(omask, spike_mask, valid_mask)
            for lst, key in [(p_list, "precision"), (r_list, "recall"), (f1_list, "f1")]:
                if not np.isnan(m[key]):
                    lst.append(m[key])
        return (
            float(np.median(p_list)) if p_list else float("nan"),
            float(np.median(r_list)) if r_list else float("nan"),
            float(np.median(f1_list)) if f1_list else float("nan"),
        )

    def test_moderate_corruption_recall(self):
        """At 30% void + σ=0.010 m: recall of large spikes must be ≥ 0.60."""
        _, recall, _ = self._run(0.30, 0.010)
        assert recall >= 0.60, f"Recall {recall:.3f} < 0.60"

    def test_moderate_corruption_precision(self):
        """At 30% void + σ=0.010 m: precision must be ≥ 0.40."""
        precision, _, _ = self._run(0.30, 0.010)
        assert precision >= 0.40, f"Precision {precision:.3f} < 0.40"

    def test_moderate_corruption_f1(self):
        """At 30% void + σ=0.010 m: F1 score must be ≥ 0.40."""
        _, _, f1 = self._run(0.30, 0.010)
        assert f1 >= 0.40, f"F1 {f1:.3f} < 0.40"

    def test_f1_reasonable_under_high_noise(self):
        """F1 must remain ≥ 0.20 even under high noise (σ=0.050 m).

        Note: the adaptive threshold scales with σ, so high-noise F1 can
        actually exceed low-noise F1 for large fixed-amplitude spikes —
        both values just need to stay above the minimum bar.
        """
        _, _, f1_low  = self._run(0.30, 0.010)
        _, _, f1_high = self._run(0.30, 0.050)
        assert f1_low  >= 0.20, f"Low-noise F1 {f1_low:.3f} < 0.20"
        assert f1_high >= 0.20, f"High-noise F1 {f1_high:.3f} < 0.20"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Inlier Retention
# ══════════════════════════════════════════════════════════════════════════════

class TestInlierRetention:
    """Clean valid pixels must not be excessively discarded."""

    @pytest.fixture(autouse=True)
    def _est(self):
        self.est = RobustDepthEstimator(outlier_k=3.0)

    @pytest.mark.parametrize("void_ratio,noise_std,min_retention", [
        (0.30, 0.010, 0.80),
        (0.50, 0.015, 0.75),
        (0.50, 0.030, 0.70),
    ])
    def test_retention_above_threshold(self, void_ratio, noise_std, min_retention):
        """Inlier retention must exceed min_retention over 8 seeds."""
        retentions = []
        for seed in range(8):
            corrupted, valid_mask, spike_mask = _with_spikes(void_ratio, noise_std, seed)
            omask, _, _ = self.est.bilateral_outlier_mask(corrupted)
            ret = inlier_retention_rate(omask, spike_mask, valid_mask)
            if not np.isnan(ret):
                retentions.append(ret)

        assert len(retentions) > 0
        assert np.median(retentions) >= min_retention, (
            f"Retention {np.median(retentions):.3f} < {min_retention} "
            f"(void={void_ratio:.0%}, σ={noise_std:.3f})"
        )

    def test_uniform_clean_map_full_retention(self):
        """On a perfectly uniform map with no spikes, nothing should be flagged."""
        d = np.full(GRID_SHAPE, TRUE_DEPTH, dtype=np.float32)
        omask, _, _ = self.est.bilateral_outlier_mask(d)
        valid  = d > 0
        spikes = np.zeros_like(valid)
        ret    = inlier_retention_rate(omask, spikes, valid)
        assert ret >= 0.99, f"Retention on clean map {ret:.4f} < 0.99"


# ══════════════════════════════════════════════════════════════════════════════
# 6. Geometry Quality (Chamfer / Hausdorff / Coverage)
# ══════════════════════════════════════════════════════════════════════════════

class TestGeometryQuality:
    """
    These tests construct a known 3-D shape, corrupt and reconstruct it,
    then measure how closely the reconstruction matches the original.
    """

    def _sphere_cloud(self, n=3000, radius=0.5, seed=0):
        rng   = np.random.default_rng(seed)
        phi   = rng.uniform(0, 2 * np.pi, n)
        ct    = rng.uniform(-1, 1, n)
        st    = np.sqrt(1 - ct ** 2)
        return np.column_stack([
            radius * st * np.cos(phi),
            radius * st * np.sin(phi),
            radius * ct,
        ])

    def test_chamfer_identical_clouds_is_zero(self):
        pts = self._sphere_cloud()
        c, h = chamfer_hausdorff(pts, pts)
        assert c < 1e-6, f"Chamfer of identical clouds: {c:.6f}"
        assert h < 1e-6, f"Hausdorff of identical clouds: {h:.6f}"

    def test_chamfer_increases_with_translation(self):
        """Translated cloud must have larger Chamfer than the original."""
        pts_a  = self._sphere_cloud()
        pts_b  = pts_a + np.array([0.5, 0, 0])
        c, _   = chamfer_hausdorff(pts_a, pts_b)
        c0, _  = chamfer_hausdorff(pts_a, pts_a)
        assert c > c0 + 0.1, (
            f"Chamfer after 0.5m translation ({c:.4f}) should exceed "
            f"self-Chamfer ({c0:.6f}) by at least 0.1"
        )

    def test_chamfer_noisy_cloud_is_small(self):
        """A cloud with small noise should have small Chamfer vs the original."""
        rng   = np.random.default_rng(0)
        pts   = self._sphere_cloud(n=2000)
        noisy = pts + rng.normal(0, 0.01, pts.shape)
        c, _  = chamfer_hausdorff(pts, noisy, n_sample=2000)
        assert c < 0.05, f"Chamfer with σ=0.01 noise: {c:.4f} ≥ 0.05"

    def test_coverage_full_on_identical(self):
        pts      = self._sphere_cloud()
        coverage = point_cloud_coverage(pts, pts, radius=0.01)
        assert coverage == pytest.approx(1.0), (
            f"Coverage of identical clouds: {coverage:.4f}"
        )

    def test_coverage_drops_with_translation(self):
        pts  = self._sphere_cloud(n=1000, radius=0.5)
        far  = pts + np.array([2.0, 0, 0])
        cov  = point_cloud_coverage(far, pts, radius=0.05)
        assert cov < 0.1, f"Coverage with 2m offset: {cov:.3f}"

    def test_coverage_partial(self):
        """Keep only one hemisphere — coverage should be roughly 0.5."""
        pts  = self._sphere_cloud(n=4000)
        half = pts[pts[:, 0] >= 0]
        cov  = point_cloud_coverage(half, pts, radius=0.05)
        assert 0.3 < cov < 0.8, f"Half-sphere coverage: {cov:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# 7. Shape Classification on Known Primitives
# ══════════════════════════════════════════════════════════════════════════════

class TestShapeClassificationMetrics:
    """Classification accuracy on analytically-constructed meshes."""

    @pytest.fixture(autouse=True)
    def _imports(self):
        import open3d as o3d
        from perception.shape_descriptor import ShapeDescriptor
        self.o3d   = o3d
        self.desc  = ShapeDescriptor

    @pytest.mark.parametrize("shape,expected_label_fragment", [
        ("sphere",    "sphere"),
        ("box",       "box"),
        ("elongated", "elongated"),
    ])
    def test_correct_label(self, shape, expected_label_fragment):
        if shape == "sphere":
            mesh = self.o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
        elif shape == "box":
            mesh = self.o3d.geometry.TriangleMesh.create_box(1, 1, 1)
        else:
            mesh = self.o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=5.0, resolution=30)

        mesh.compute_vertex_normals()
        d     = self.desc.describe(mesh)
        label = self.desc.classify(d)
        assert expected_label_fragment.lower() in label.lower(), (
            f"Shape '{shape}' classified as '{label}', "
            f"expected label containing '{expected_label_fragment}'"
        )

    def test_sphere_compactness_near_one(self):
        mesh = self.o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
        d    = self.desc.describe(mesh)
        assert d["compactness"] > 0.85, (
            f"Sphere compactness {d['compactness']:.4f} ≤ 0.85"
        )

    def test_elongated_aspect_ratio_large(self):
        mesh = self.o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=5.0, resolution=30)
        d    = self.desc.describe(mesh)
        assert d["aspect_ratio"] > 3.0, (
            f"Elongated shape aspect ratio {d['aspect_ratio']:.2f} ≤ 3.0"
        )

    def test_cube_aspect_ratio_near_one(self):
        mesh = self.o3d.geometry.TriangleMesh.create_box(1, 1, 1)
        d    = self.desc.describe(mesh)
        assert d["aspect_ratio"] < 1.5, (
            f"Cube aspect ratio {d['aspect_ratio']:.2f} ≥ 1.5"
        )

    def test_classification_accuracy_at_least_80pct(self):
        """At least 80% of test primitives must get the correct label."""
        cases = [
            (self.o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30), "sphere"),
            (self.o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=25), "sphere"),
            (self.o3d.geometry.TriangleMesh.create_box(1, 1, 1), "box"),
            (self.o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=5.0, resolution=30), "elongated"),
            (self.o3d.geometry.TriangleMesh.create_cylinder(radius=0.2, height=4.0, resolution=30), "elongated"),
        ]
        correct = 0
        for mesh, frag in cases:
            mesh.compute_vertex_normals()
            label = self.desc.classify(self.desc.describe(mesh))
            if frag.lower() in label.lower():
                correct += 1
        acc = correct / len(cases)
        assert acc >= 0.80, f"Shape classification accuracy {acc:.0%} < 80%"


# ══════════════════════════════════════════════════════════════════════════════
# 8. Full Pipeline Metrics (flat synthetic scene, no external data)
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineMetrics:
    """
    Run the entire pipeline on a synthetic flat depth map and assert
    end-to-end metric bounds.  No Bunny data files required.
    """

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        from pipeline import Pipeline

        class _FlatStream:
            """
            Minimal stream: bowl-shaped depth map centred at TRUE_DEPTH.

            A perfectly flat (coplanar) point cloud causes the alpha-shape
            reconstructor to crash, so we add a gentle paraboloid to ensure
            the back-projected points span all three spatial dimensions.
            """
            def get_frame(self):
                rng = np.random.default_rng(42)
                H, W = GRID_SHAPE
                u = np.linspace(-1, 1, W, dtype=np.float32)
                v = np.linspace(-1, 1, H, dtype=np.float32)
                U, V  = np.meshgrid(u, v)
                # paraboloid: depth varies ±0.15 m across the frame
                depth = (TRUE_DEPTH + 0.15 * (U ** 2 + V ** 2)).astype(np.float32)
                depth += rng.normal(0, 0.010, GRID_SHAPE).astype(np.float32)
                np.clip(depth, 0, None, out=depth)
                # pinhole intrinsics for a 100×100 virtual camera
                fx = fy = 80.0
                cx, cy = W / 2.0, H / 2.0
                return {
                    "depth":      depth,
                    "intrinsics": (fx, fy, cx, cy),
                    "rgb":        None,
                }

        def corrupt(d):
            rng  = np.random.default_rng(7)
            out  = d.copy()
            vi   = np.flatnonzero(out > 0)
            n    = int(len(vi) * 0.50)
            out.flat[rng.choice(vi, n, replace=False)] = 0.0
            still = out > 0
            out[still] += rng.normal(0, 0.015, still.sum()).astype(np.float32)
            np.clip(out, 0, None, out=out)
            return out

        return Pipeline(_FlatStream()).run(corrupt_fn=corrupt)

    def test_z_est_within_15cm_of_true(self, pipeline_result):
        """
        z_est must stay within 15 cm of z_true (the pipeline's own reference).

        The scene is a paraboloid, not a flat plane, so the absolute offset
        from TRUE_DEPTH is irrelevant — what matters is that the filter's
        estimate tracks the uncorrupted median depth closely.
        """
        assert pipeline_result.z_est  is not None, "z_est is None"
        assert pipeline_result.z_true is not None, "z_true is None"
        err = z_error_cm(pipeline_result.z_est, pipeline_result.z_true)
        assert err < 15.0, f"Z error vs z_true: {err:.3f} cm ≥ 15 cm"

    def test_clean_depth_has_valid_pixels(self, pipeline_result):
        n_clean = int((pipeline_result.clean_depth > 0).sum())
        assert n_clean > 100, f"Only {n_clean} clean pixels"

    def test_sigma_map_is_positive(self, pipeline_result):
        smap  = pipeline_result.sigma_map
        valid = pipeline_result.raw_depth > 0
        assert smap is not None
        assert (smap[valid] > 0).all(), "sigma_map has non-positive values at valid pixels"

    def test_point_cloud_nonempty(self, pipeline_result):
        assert pipeline_result.point_cloud is not None
        assert len(pipeline_result.point_cloud) > 10

    def test_embedding_shape(self, pipeline_result):
        emb = pipeline_result.embedding
        assert emb is not None
        assert emb.shape == (256,), f"Expected (256,), got {emb.shape}"

    def test_embedding_finite(self, pipeline_result):
        assert np.all(np.isfinite(pipeline_result.embedding)), \
            "Embedding contains non-finite values"

    def test_shape_label_is_string(self, pipeline_result):
        assert isinstance(pipeline_result.shape_label, str)
        assert len(pipeline_result.shape_label) > 0

    def test_timings_all_positive(self, pipeline_result):
        for stage, t in pipeline_result.timings.items():
            assert t > 0, f"Stage '{stage}' timing {t} ≤ 0"

    def test_timings_total_reasonable(self, pipeline_result):
        """Full pipeline on a 100×100 synthetic map should finish in < 60 s."""
        total = pipeline_result.timings.get("total", float("inf"))
        assert total < 60.0, f"Pipeline took {total:.1f} s — unexpectedly slow"

    def test_per_point_features_shape(self, pipeline_result):
        pp = pipeline_result.per_point_features
        assert pp is not None
        assert pp.ndim == 2
        assert pp.shape[1] == 1024, f"Expected (N, 1024), got {pp.shape}"
