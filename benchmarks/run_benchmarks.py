"""
Engine Grounder — Comprehensive Benchmark Runner
=================================================

Sections
--------
  A. Depth-filter parameter sweep  (void_ratio × noise_std grid, 5 seeds each)
  B. Outlier-detection quality     (controlled spike injection, P/R/F1)
  C. Full pipeline on Stanford Bunny (requires data/bunny_depth.npy)
  D. Shape-classification accuracy  (sphere, box, elongated cylinder)
  E. Per-stage timing profiling     (N_TIMING_RUNS repetitions)

Outputs
-------
  - Formatted console tables for every section
  - JSON dump:  benchmarks/results.json
  - HTML report: benchmarks/benchmark_report.html  (Plotly heatmaps + bar charts)

Usage
-----
  # from engine_grounder/
  python benchmarks/run_benchmarks.py
  python benchmarks/run_benchmarks.py --skip-bunny   # skip Bunny (no data files needed)
  python benchmarks/run_benchmarks.py --out my_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from engine_grounder.geometry.depth_filter import RobustDepthEstimator
from benchmarks.metrics import (
    z_error_cm, depth_rmse, depth_mae, depth_bias,
    rmse_improvement_ratio, outlier_metrics, inlier_retention_rate,
    sigma_recovery_rel_error, chamfer_hausdorff, point_cloud_coverage,
)

# ══════════════════════════════════════════════════════════════════════════════
# Benchmark parameters
# ══════════════════════════════════════════════════════════════════════════════

VOID_RATIOS   = [0.10, 0.30, 0.50, 0.70, 0.85]
NOISE_STDS    = [0.005, 0.015, 0.030, 0.050]
N_SEEDS       = 5       # Monte-Carlo seeds per grid cell
GRID_SIZE     = (100, 100)
TRUE_DEPTH    = 2.0     # metres

N_SPIKES      = 60      # controlled outlier-detection experiment
SPIKE_AMP     = 0.6     # metres above mean

N_TIMING_RUNS = 5       # repetitions for timing profiling

BUNNY_VOID    = 0.55
BUNNY_NOISE   = 0.015

# ══════════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v, fmt=".3f", na="  N/A "):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na
    return format(v, fmt)


def _hline(widths):
    return "+" + "+".join("-" * (w + 2) for w in widths) + "+"


def _row(cells, widths):
    parts = []
    for c, w in zip(cells, widths):
        parts.append(f" {str(c):<{w}} ")
    return "|" + "|".join(parts) + "|"


def _table(headers, rows, title=""):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    lines = []
    if title:
        total = sum(w + 3 for w in widths) + 1
        lines.append(f"\n{'─'*total}")
        lines.append(f"  {title}")
        lines.append('─'*total)
    lines.append(_hline(widths))
    lines.append(_row(headers, widths))
    lines.append(_hline(widths))
    for r in rows:
        lines.append(_row(r, widths))
    lines.append(_hline(widths))
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_corrupted(
    true_depth: float,
    shape: tuple,
    void_ratio: float,
    noise_std: float,
    seed: int,
):
    """Return (ground_truth, corrupted, true_void_mask)."""
    rng  = np.random.default_rng(seed)
    gt   = np.full(shape, true_depth, dtype=np.float32)
    noisy = gt + rng.normal(0, noise_std, shape).astype(np.float32)
    np.clip(noisy, 0.0, None, out=noisy)

    # inject voids
    n_void = int(noisy.size * void_ratio)
    void_idx = rng.choice(noisy.size, n_void, replace=False)
    noisy.flat[void_idx] = 0.0
    void_mask = np.zeros(shape, dtype=bool)
    void_mask.flat[void_idx] = True

    return gt, noisy, void_mask


# ══════════════════════════════════════════════════════════════════════════════
# Section A — Depth-filter grid sweep
# ══════════════════════════════════════════════════════════════════════════════

def run_filter_sweep() -> List[dict]:
    """Return list of per-cell metric dicts."""
    estimator = RobustDepthEstimator(outlier_k=3.0, tau_threshold=0.10)
    records   = []

    total = len(VOID_RATIOS) * len(NOISE_STDS) * N_SEEDS
    done  = 0

    print("\n[A] Depth-filter grid sweep  "
          f"({len(VOID_RATIOS)} void ratios × {len(NOISE_STDS)} noise levels"
          f" × {N_SEEDS} seeds = {total} runs) ...")

    for vr in VOID_RATIOS:
        for ns in NOISE_STDS:
            cell_z_err   = []
            cell_rmse_b  = []
            cell_rmse_a  = []
            cell_mae_a   = []
            cell_bias_a  = []
            cell_sig_err = []
            cell_flag    = []
            cell_timing  = []

            for seed in range(N_SEEDS):
                gt, corrupted, _ = _make_corrupted(
                    TRUE_DEPTH, GRID_SIZE, vr, ns, seed
                )
                valid_mask = corrupted > 0

                t0 = time.perf_counter()
                omask, sigma_map, _ = estimator.bilateral_outlier_mask(corrupted)
                z_est = estimator.get_stable_z(corrupted)
                elapsed = time.perf_counter() - t0

                clean = corrupted.copy()
                clean[omask] = 0.0
                clean_valid = (clean > 0)

                rmse_b = depth_rmse(corrupted, gt, mask=valid_mask)
                rmse_a = depth_rmse(clean,     gt, mask=clean_valid)
                mae_a  = depth_mae(clean,      gt, mask=clean_valid)
                bias_a = depth_bias(clean,     gt, mask=clean_valid)

                if z_est is not None:
                    cell_z_err.append(z_error_cm(z_est, TRUE_DEPTH))

                if sigma_map is not None and valid_mask.any():
                    sigma_hat = float(np.median(sigma_map[valid_mask]))
                    cell_sig_err.append(
                        sigma_recovery_rel_error(sigma_hat, ns)
                    )

                n_valid = int(valid_mask.sum())
                n_flag  = int((omask & valid_mask).sum())
                cell_flag.append(n_flag / n_valid if n_valid else float("nan"))

                cell_rmse_b.append(rmse_b)
                cell_rmse_a.append(rmse_a)
                cell_mae_a.append(mae_a)
                cell_bias_a.append(bias_a)
                cell_timing.append(elapsed * 1000)
                done += 1

            def med(lst):
                arr = [x for x in lst if x is not None and not np.isnan(x)]
                return float(np.median(arr)) if arr else float("nan")

            rmse_b_med = med(cell_rmse_b)
            rmse_a_med = med(cell_rmse_a)

            records.append({
                "void_ratio":     vr,
                "noise_std":      ns,
                "z_error_cm":     med(cell_z_err),
                "rmse_before":    rmse_b_med,
                "rmse_after":     rmse_a_med,
                "rmse_ratio":     rmse_improvement_ratio(rmse_b_med, rmse_a_med),
                "mae_after":      med(cell_mae_a),
                "bias_after":     med(cell_bias_a),
                "sigma_rel_err":  med(cell_sig_err),
                "flag_rate":      med(cell_flag),
                "timing_ms":      med(cell_timing),
            })

    return records


def print_filter_sweep(records: List[dict]):
    print("\n── A. Z Estimation Error (cm) ──")
    _print_grid(records, "z_error_cm", fmt=".2f")

    print("\n── A. RMSE Improvement Ratio (before/after) ──")
    _print_grid(records, "rmse_ratio", fmt=".2f")

    print("\n── A. σ̂ Recovery Relative Error ──")
    _print_grid(records, "sigma_rel_err", fmt=".2f")

    print("\n── A. Flag Rate (fraction of valid pixels flagged) ──")
    _print_grid(records, "flag_rate", fmt=".3f")

    print("\n── A. Median Filter Time (ms, 100×100 map) ──")
    _print_grid(records, "timing_ms", fmt=".1f")


def _print_grid(records: List[dict], key: str, fmt: str):
    ns_vals = sorted(set(r["noise_std"] for r in records))
    vr_vals = sorted(set(r["void_ratio"] for r in records))
    lookup  = {(r["void_ratio"], r["noise_std"]): r[key] for r in records}

    headers = ["void \\ noise"] + [f"σ={ns:.3f}" for ns in ns_vals]
    rows    = []
    for vr in vr_vals:
        row = [f"{vr:.0%}"]
        for ns in ns_vals:
            v = lookup.get((vr, ns))
            row.append(_fmt(v, fmt))
        rows.append(row)

    print(_table(headers, rows))


# ══════════════════════════════════════════════════════════════════════════════
# Section B — Outlier detection quality (controlled spike injection)
# ══════════════════════════════════════════════════════════════════════════════

def run_outlier_detection_quality() -> List[dict]:
    """
    For each (void_ratio, noise_std) inject N_SPIKES known large-amplitude
    spikes, run the filter, measure precision / recall / F1.
    """
    print(f"\n[B] Outlier-detection quality  "
          f"({N_SPIKES} injected spikes, amp={SPIKE_AMP} m) ...")

    params = [
        (0.30, 0.010),
        (0.30, 0.020),
        (0.50, 0.010),
        (0.50, 0.020),
        (0.50, 0.030),
        (0.70, 0.010),
        (0.70, 0.030),
    ]

    estimator = RobustDepthEstimator(outlier_k=3.0)
    records   = []

    for vr, ns in params:
        p_list, r_list, f1_list, ret_list = [], [], [], []

        for seed in range(N_SEEDS):
            rng  = np.random.default_rng(seed + 100)
            gt, corrupted, void_mask = _make_corrupted(
                TRUE_DEPTH, GRID_SIZE, vr, ns, seed
            )
            valid_mask = corrupted > 0

            # inject known spikes into valid pixels
            valid_idx = np.flatnonzero(valid_mask & ~void_mask)
            if len(valid_idx) < N_SPIKES:
                continue
            spike_idx = rng.choice(valid_idx, N_SPIKES, replace=False)
            corrupted.flat[spike_idx] += float(SPIKE_AMP)
            true_spike_mask = np.zeros(GRID_SIZE, dtype=bool)
            true_spike_mask.flat[spike_idx] = True

            omask, _, _ = estimator.bilateral_outlier_mask(corrupted)

            m   = outlier_metrics(omask, true_spike_mask, valid_mask)
            ret = inlier_retention_rate(omask, true_spike_mask, valid_mask)

            p_list.append(m["precision"])
            r_list.append(m["recall"])
            f1_list.append(m["f1"])
            ret_list.append(ret)

        def med(lst):
            arr = [x for x in lst if x is not None and not np.isnan(x)]
            return float(np.median(arr)) if arr else float("nan")

        records.append({
            "void_ratio":   vr,
            "noise_std":    ns,
            "precision":    med(p_list),
            "recall":       med(r_list),
            "f1":           med(f1_list),
            "retention":    med(ret_list),
        })

    return records


def print_outlier_quality(records: List[dict]):
    headers = ["Void%", "Noise σ", "Precision", "Recall", "F1", "Inlier Retention"]
    rows    = [
        [
            f"{r['void_ratio']:.0%}",
            f"{r['noise_std']:.3f} m",
            _fmt(r["precision"], ".3f"),
            _fmt(r["recall"],    ".3f"),
            _fmt(r["f1"],        ".3f"),
            _fmt(r["retention"], ".3f"),
        ]
        for r in records
    ]
    print(_table(headers, rows, title="B. Outlier Detection Quality (controlled spike injection)"))


# ══════════════════════════════════════════════════════════════════════════════
# Section C — Full pipeline on Stanford Bunny
# ══════════════════════════════════════════════════════════════════════════════

def run_bunny_pipeline() -> dict:
    """Run the full 6-stage pipeline and collect all metrics."""
    print(f"\n[C] Full pipeline on Stanford Bunny  "
          f"(void={BUNNY_VOID:.0%}, noise={BUNNY_NOISE:.3f} m) ...")

    from engine_grounder.streams.mock_stream import BunnyStream
    from engine_grounder.pipeline import Pipeline, PipelineResult

    rng = np.random.default_rng(7)

    def inject(depth):
        out = depth.copy()
        valid_idx = np.flatnonzero(out > 0)
        n_void    = int(len(valid_idx) * BUNNY_VOID)
        out.flat[rng.choice(valid_idx, n_void, replace=False)] = 0.0
        still = out > 0
        out[still] += rng.normal(0, BUNNY_NOISE, still.sum()).astype(np.float32)
        np.clip(out, 0.0, None, out=out)
        return out

    data_dir = os.path.join(_HERE, "..", "data")
    stream = BunnyStream(
        depth_path=os.path.join(data_dir, "bunny_depth.npy"),
        intrinsics_path=os.path.join(data_dir, "bunny_intrinsics.npy"),
    )
    pipe   = Pipeline(stream, outlier_k=3.0, tau=0.10)
    result = pipe.run(corrupt_fn=inject)

    rd    = result.raw_depth
    clean = result.clean_depth
    omask = result.outlier_mask
    smap  = result.sigma_map

    n_total = rd.size
    n_valid = int((rd > 0).sum())
    n_out   = int(omask.sum()) if omask is not None else 0
    n_clean = int((clean > 0).sum()) if clean is not None else 0

    void_pct    = (n_total - n_valid) / n_total * 100
    out_pct     = n_out / n_valid * 100 if n_valid else float("nan")
    retention   = n_clean / n_valid if n_valid else float("nan")

    sigma_hat   = float(np.median(smap[(rd > 0)])) if smap is not None and (rd > 0).any() else float("nan")
    sig_rel_err = sigma_recovery_rel_error(sigma_hat, BUNNY_NOISE)

    z_err = None
    if result.z_est is not None and result.z_true is not None:
        z_err = z_error_cm(result.z_est, result.z_true)

    # Chamfer distance: reference = original clean point cloud,
    # reconstructed  = filtered point cloud
    chamfer, hausdorff = float("nan"), float("nan")
    coverage           = float("nan")
    if result.point_cloud is not None and len(result.point_cloud) > 0:
        # Build reference cloud from unfiltered valid pixels
        raw_copy   = rd.copy()
        raw_copy[rd <= 0] = 0
        intr       = result.intrinsics
        if intr is not None:
            fx, fy, cx, cy = intr
            h, w    = rd.shape
            U, V    = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
            Z       = rd.ravel()
            valid_r = Z > 0
            X = (U.ravel()[valid_r] - cx) * Z[valid_r] / fx
            Y = (V.ravel()[valid_r] - cy) * Z[valid_r] / fy
            ref_pts = np.column_stack([X, Y, Z[valid_r]])
            chamfer, hausdorff = chamfer_hausdorff(
                result.point_cloud, ref_pts, n_sample=5_000
            )
            coverage = point_cloud_coverage(
                result.point_cloud, ref_pts, radius=0.05
            )

    emb_norm = float(np.linalg.norm(result.embedding)) if result.embedding is not None else float("nan")

    n_mesh_v = len(result.mesh.vertices)   if result.mesh            else 0
    n_mesh_f = len(result.mesh.triangles)  if result.mesh            else 0
    n_simp_v = len(result.mesh_simplified.vertices)  if result.mesh_simplified else 0
    n_simp_f = len(result.mesh_simplified.triangles) if result.mesh_simplified else 0

    return {
        "depth_map_shape":    f"{rd.shape[1]}×{rd.shape[0]}",
        "total_pixels":       n_total,
        "void_pct":           void_pct,
        "valid_after_corrupt": n_valid,
        "outliers_flagged":   n_out,
        "outlier_pct":        out_pct,
        "clean_pixels":       n_clean,
        "inlier_retention":   retention,
        "sigma_hat":          sigma_hat,
        "sigma_rel_err":      sig_rel_err,
        "z_est":              result.z_est,
        "z_true":             result.z_true,
        "z_error_cm":         z_err,
        "point_cloud_pts":    len(result.point_cloud) if result.point_cloud is not None else 0,
        "chamfer_m":          chamfer,
        "hausdorff_m":        hausdorff,
        "surface_coverage":   coverage,
        "mesh_vertices":      n_mesh_v,
        "mesh_faces":         n_mesh_f,
        "simplified_vertices": n_simp_v,
        "simplified_faces":   n_simp_f,
        "embedding_dim":      result.embedding.shape[0] if result.embedding is not None else 0,
        "embedding_l2":       emb_norm,
        "shape_label":        result.shape_label,
        "compactness":        result.shape_descriptors.get("compactness", float("nan")) if result.shape_descriptors else float("nan"),
        "aspect_ratio":       result.shape_descriptors.get("aspect_ratio", float("nan")) if result.shape_descriptors else float("nan"),
        "timings":            result.timings,
    }


def print_bunny_pipeline(r: dict):
    title = "C. Full Pipeline — Stanford Bunny"
    total = sum(r["timings"].values()) * 1000 if r.get("timings") else float("nan")

    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    print(f"  Depth map        : {r['depth_map_shape']} ({r['total_pixels']:,} px)")
    print(f"  Void injected    : {r['void_pct']:.1f}%  →  "
          f"{r['valid_after_corrupt']:,} valid px remain")
    print(f"  Outliers flagged : {r['outliers_flagged']:,}  "
          f"({_fmt(r['outlier_pct'], '.1f')}% of valid)")
    print(f"  Inlier retention : {_fmt(r['inlier_retention'], '.3f')}")
    print(f"  σ̂ estimated     : {_fmt(r['sigma_hat'], '.4f')} m  "
          f"(true={BUNNY_NOISE:.3f} m,  rel.err={_fmt(r['sigma_rel_err'], '.2f')})")
    print(f"  Z_est / Z_true   : {_fmt(r['z_est'], '.4f')} m  /  "
          f"{_fmt(r['z_true'], '.4f')} m   →  "
          f"error = {_fmt(r['z_error_cm'], '.3f')} cm")
    print()
    print(f"  Point cloud      : {r['point_cloud_pts']:,} pts")
    print(f"  Chamfer dist.    : {_fmt(r['chamfer_m'], '.4f')} m")
    print(f"  Hausdorff dist.  : {_fmt(r['hausdorff_m'], '.4f')} m")
    print(f"  Surface coverage : {_fmt(r['surface_coverage'], '.3f')} "
          "(fraction within 5 cm)")
    print()
    print(f"  Mesh (full)      : {r['mesh_vertices']:,} verts, {r['mesh_faces']:,} faces")
    print(f"  Mesh (simple)    : {r['simplified_vertices']:,} verts, {r['simplified_faces']:,} faces")
    print()
    print(f"  Embedding        : {r['embedding_dim']}-d,  L2={_fmt(r['embedding_l2'], '.4f')}")
    print(f"  Shape label      : {r['shape_label']}")
    print(f"  Compactness      : {_fmt(r['compactness'], '.4f')}")
    print(f"  Aspect ratio     : {_fmt(r['aspect_ratio'], '.2f')}")
    print()
    print(f"  ── Timings ──")
    for stage, t in r["timings"].items():
        print(f"     {stage:<14}: {t*1000:>7.0f} ms")
    print(f"     {'TOTAL':<14}: {total:>7.0f} ms")


# ══════════════════════════════════════════════════════════════════════════════
# Section D — Shape classification accuracy
# ══════════════════════════════════════════════════════════════════════════════

def run_shape_classification() -> List[dict]:
    print("\n[D] Shape-classification accuracy ...")

    import open3d as o3d
    from engine_grounder.perception.shape_descriptor import ShapeDescriptor

    shapes = [
        ("sphere r=1.0",   o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30),   "sphere-like"),
        ("sphere r=0.5",   o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=30),   "sphere-like"),
        ("cube 1×1×1",     o3d.geometry.TriangleMesh.create_box(1, 1, 1),                        "box-like / compact"),
        ("box 1×2×1",      o3d.geometry.TriangleMesh.create_box(1, 2, 1),                        "box-like / compact"),
        ("cylinder r=0.1 h=5", o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=5.0, resolution=30), "elongated"),
        ("cylinder r=0.2 h=4", o3d.geometry.TriangleMesh.create_cylinder(radius=0.2, height=4.0, resolution=30), "elongated"),
    ]

    records = []
    for name, mesh, expected_contains in shapes:
        mesh.compute_vertex_normals()
        desc  = ShapeDescriptor.describe(mesh)
        label = ShapeDescriptor.classify(desc)
        correct = expected_contains.lower() in label.lower()
        records.append({
            "shape":       name,
            "expected":    expected_contains,
            "predicted":   label,
            "correct":     correct,
            "compactness": desc.get("compactness", float("nan")),
            "aspect_ratio": desc.get("aspect_ratio", float("nan")),
        })

    return records


def print_shape_classification(records: List[dict]):
    headers = ["Shape", "Expected (contains)", "Predicted", "Correct?", "Compact.", "AR"]
    rows    = [
        [
            r["shape"],
            r["expected"],
            r["predicted"],
            "✓" if r["correct"] else "✗",
            _fmt(r["compactness"],  ".3f"),
            _fmt(r["aspect_ratio"], ".2f"),
        ]
        for r in records
    ]
    acc = sum(r["correct"] for r in records) / len(records) * 100
    print(_table(headers, rows, title="D. Shape Classification Accuracy"))
    print(f"  Accuracy: {acc:.0f}%  ({sum(r['correct'] for r in records)}/{len(records)} correct)")


# ══════════════════════════════════════════════════════════════════════════════
# Section E — Per-stage timing profiling
# ══════════════════════════════════════════════════════════════════════════════

def run_timing_profile(skip_bunny: bool) -> dict:
    print(f"\n[E] Per-stage timing profiling ({N_TIMING_RUNS} runs) ...")

    if skip_bunny:
        print("  (skipped — pass --no-skip-bunny to enable)")
        return {}

    from engine_grounder.streams.mock_stream import BunnyStream
    from engine_grounder.pipeline import Pipeline

    data_dir = os.path.join(_HERE, "..", "data")
    rng      = np.random.default_rng(99)

    def corrupt(d):
        out = d.copy()
        vi  = np.flatnonzero(out > 0)
        n   = int(len(vi) * BUNNY_VOID)
        out.flat[rng.choice(vi, n, replace=False)] = 0.0
        still = out > 0
        out[still] += rng.normal(0, BUNNY_NOISE, still.sum()).astype(np.float32)
        np.clip(out, 0, None, out=out)
        return out

    stage_times: Dict[str, List[float]] = {}

    for i in range(N_TIMING_RUNS):
        stream = BunnyStream(
            depth_path      = os.path.join(data_dir, "bunny_depth.npy"),
            intrinsics_path = os.path.join(data_dir, "bunny_intrinsics.npy"),
        )
        result = Pipeline(stream).run(corrupt_fn=corrupt)
        for stage, t in result.timings.items():
            stage_times.setdefault(stage, []).append(t * 1000)

    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in stage_times.items()}


def print_timing_profile(timing: dict):
    if not timing:
        return
    headers = ["Stage", "Mean (ms)", "Std (ms)", "% of total"]
    total_mean = sum(v["mean"] for k, v in timing.items() if k != "total")
    rows = []
    for stage, v in timing.items():
        pct = v["mean"] / total_mean * 100 if stage != "total" and total_mean else float("nan")
        rows.append([
            stage,
            _fmt(v["mean"], ".1f"),
            _fmt(v["std"],  ".1f"),
            _fmt(pct,       ".1f"),
        ])
    print(_table(headers, rows, title="E. Per-stage Timing Profile"))


# ══════════════════════════════════════════════════════════════════════════════
# HTML report
# ══════════════════════════════════════════════════════════════════════════════

def build_html_report(
    filter_records: List[dict],
    outlier_records: List[dict],
    shape_records: List[dict],
    timing: dict,
    bunny: dict,
    output_path: str,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    vr_vals = sorted(set(r["void_ratio"] for r in filter_records))
    ns_vals = sorted(set(r["noise_std"]  for r in filter_records))
    lookup  = {(r["void_ratio"], r["noise_std"]): r for r in filter_records}

    def _grid(key):
        return np.array(
            [[lookup[(vr, ns)][key] for ns in ns_vals] for vr in vr_vals],
            dtype=float,
        )

    x_labels = [f"σ={ns:.3f}" for ns in ns_vals]
    y_labels  = [f"{vr:.0%}"  for vr in vr_vals]

    def _heatmap(z, title, colorscale, zmin=None, zmax=None):
        return go.Heatmap(
            z=z, x=x_labels, y=y_labels,
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            text=np.round(z, 3).astype(str),
            texttemplate="%{text}",
            hovertemplate="void=%{y}  noise=%{x}<br>value=%{z:.3f}<extra></extra>",
            colorbar=dict(thickness=10, len=0.4),
        )

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            "Z Error (cm)",
            "RMSE Improvement Ratio",
            "σ̂ Recovery Relative Error",
            "Flag Rate",
            "Filter Time (ms)",
            "Outlier Detection Metrics",
            "Shape Classification",
            "Per-stage Timing",
            "Bunny Pipeline Summary",
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "domain"}],
        ],
    )

    # Row 1
    fig.add_trace(_heatmap(_grid("z_error_cm"),    "Z err cm",   "RdYlGn_r"), row=1, col=1)
    fig.add_trace(_heatmap(_grid("rmse_ratio"),     "RMSE ratio", "Greens",   zmin=0), row=1, col=2)
    fig.add_trace(_heatmap(_grid("sigma_rel_err"),  "σ rel err",  "RdYlGn_r", zmin=0, zmax=1), row=1, col=3)

    # Row 2
    fig.add_trace(_heatmap(_grid("flag_rate"),   "flag rate", "Blues"), row=2, col=1)
    fig.add_trace(_heatmap(_grid("timing_ms"),   "ms",        "Purples"), row=2, col=2)

    # Outlier bar chart
    if outlier_records:
        labels = [f"v={r['void_ratio']:.0%}/σ={r['noise_std']:.2f}" for r in outlier_records]
        fig.add_trace(go.Bar(name="Precision", x=labels,
                             y=[r["precision"] for r in outlier_records],
                             marker_color="#3498db"), row=2, col=3)
        fig.add_trace(go.Bar(name="Recall",    x=labels,
                             y=[r["recall"]    for r in outlier_records],
                             marker_color="#e74c3c"), row=2, col=3)
        fig.add_trace(go.Bar(name="F1",        x=labels,
                             y=[r["f1"]        for r in outlier_records],
                             marker_color="#2ecc71"), row=2, col=3)
        fig.update_layout(barmode="group")

    # Shape classification bar
    if shape_records:
        names   = [r["shape"] for r in shape_records]
        correct = [1 if r["correct"] else 0 for r in shape_records]
        colors  = ["#2ecc71" if c else "#e74c3c" for c in correct]
        fig.add_trace(go.Bar(x=names, y=correct, marker_color=colors,
                             showlegend=False), row=3, col=1)
        fig.update_yaxes(range=[0, 1.2], row=3, col=1)

    # Timing bar
    if timing:
        stages = [k for k in timing if k != "total"]
        means  = [timing[k]["mean"] for k in stages]
        stds   = [timing[k]["std"]  for k in stages]
        fig.add_trace(go.Bar(x=stages, y=means,
                             error_y=dict(type="data", array=stds, visible=True),
                             marker_color="#9b59b6", showlegend=False), row=3, col=2)

    # Bunny summary table
    if bunny:
        t_keys = [
            "z_error_cm", "inlier_retention", "sigma_rel_err",
            "chamfer_m", "surface_coverage", "shape_label",
            "compactness", "aspect_ratio",
        ]
        t_labels = [
            "Z error (cm)", "Inlier retention", "σ̂ rel. error",
            "Chamfer (m)", "Coverage @5cm", "Shape label",
            "Compactness", "Aspect ratio",
        ]
        t_vals = [_fmt(bunny.get(k), ".4f") if isinstance(bunny.get(k), float)
                  else str(bunny.get(k, "N/A")) for k in t_keys]
        fig.add_trace(go.Table(
            header=dict(values=["Metric", "Value"],
                        fill_color="#1a1a1a",
                        font=dict(color="white", size=11)),
            cells=dict(values=[t_labels, t_vals],
                       fill_color="#111",
                       font=dict(color="white", size=10),
                       height=22),
        ), row=3, col=3)

    fig.update_layout(
        title=dict(text="Engine Grounder — Benchmark Report",
                   font=dict(size=16, color="white"), x=0.5),
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        font=dict(color="white"),
        height=1400,
        margin=dict(l=20, r=20, t=80, b=20),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="white")

    fig.write_html(output_path, auto_open=False)
    print(f"\n-> HTML report saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Engine Grounder benchmark runner")
    parser.add_argument("--skip-bunny", action="store_true",
                        help="Skip the full Bunny pipeline (no data files needed)")
    parser.add_argument("--out", default=None,
                        help="JSON output path (default: benchmarks/results.json)")
    args = parser.parse_args()

    out_json = args.out or os.path.join(_HERE, "results.json")
    out_html = os.path.splitext(out_json)[0] + "_report.html"

    print("=" * 60)
    print("  Engine Grounder — Benchmark Suite")
    print("=" * 60)

    # A — filter grid sweep
    filter_records = run_filter_sweep()
    print_filter_sweep(filter_records)

    # B — outlier detection quality
    outlier_records = run_outlier_detection_quality()
    print_outlier_quality(outlier_records)

    # C — full pipeline
    bunny_result = {}
    if not args.skip_bunny:
        data_dir = os.path.join(_HERE, "..", "data")
        bunny_npy = os.path.join(data_dir, "bunny_depth.npy")
        if not os.path.isfile(bunny_npy):
            print(f"\n[C] Skipping Bunny — data not found at {bunny_npy}")
            print("    Run  python prepare_bunny.py  once to generate it.")
        else:
            try:
                bunny_result = run_bunny_pipeline()
                print_bunny_pipeline(bunny_result)
            except Exception as exc:
                print(f"\n[C] Bunny pipeline failed: {exc}")
    else:
        print("\n[C] Skipped (--skip-bunny)")

    # D — shape classification
    shape_records = run_shape_classification()
    print_shape_classification(shape_records)

    # E — timing
    timing = run_timing_profile(skip_bunny=args.skip_bunny or not bunny_result)
    print_timing_profile(timing)

    # Save JSON
    results = {
        "filter_sweep":       filter_records,
        "outlier_quality":    outlier_records,
        "bunny_pipeline":     bunny_result,
        "shape_classification": shape_records,
        "timing_profile":     timing,
    }
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\n-> JSON results saved → {out_json}")

    # HTML report
    try:
        build_html_report(
            filter_records, outlier_records, shape_records, timing, bunny_result,
            output_path=out_html,
        )
    except Exception as exc:
        print(f"  (HTML report skipped: {exc})")

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
