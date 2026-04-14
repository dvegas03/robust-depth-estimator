"""
NAIRR Sprint 3 — Full Pipeline Demo on Stanford Bunny.

Stages:
  0. Ingest depth + intrinsics (BunnyStream)
  1. Inject synthetic corruption (voids + noise)
  2. Adaptive bilateral-residual depth filtering
  3. Pinhole back-projection → 3-D point cloud
  4. Poisson mesh reconstruction
  5. PointNet shape encoding → 256-d embedding
  6. Geometric shape descriptors + classification
  7. Interactive 6-panel Plotly HTML

Run from the repo root after ``pip install -e .``:
    OMP_NUM_THREADS=1 python examples/main.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

from engine_grounder.streams.mock_stream import BunnyStream
from engine_grounder.pipeline import Pipeline, PipelineResult
from engine_grounder.spatial.visualizer import render_pipeline_demo

OUT_HTML = "pipeline_demo.html"
VOID_RATIO = 0.55
NOISE_STD = 0.015
OUTLIER_K = 3.0
TAU = 0.10


def inject_corruption(depth: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(7)
    out = depth.copy()
    valid_idx = np.flatnonzero(out > 0)
    n_void = int(len(valid_idx) * VOID_RATIO)
    out.flat[rng.choice(valid_idx, n_void, replace=False)] = 0.0
    still = out > 0
    out[still] += rng.normal(0, NOISE_STD, still.sum()).astype(np.float32)
    np.clip(out, 0.0, None, out=out)
    return out


def main():
    print("=== Robust Depth Estimator — Full Pipeline Demo ===\n")

    stream = BunnyStream()
    pipe = Pipeline(stream, outlier_k=OUTLIER_K, tau=TAU)

    print("  Running pipeline (6 stages) ...\n")
    result: PipelineResult = pipe.run(corrupt_fn=inject_corruption)

    # ── Console report ────────────────────────────────────────────────────────
    rd = result.raw_depth
    n_total = rd.size
    n_valid = int((rd > 0).sum())
    n_out = int(result.outlier_mask.sum()) if result.outlier_mask is not None else 0
    n_clean = int((result.clean_depth > 0).sum()) if result.clean_depth is not None else 0

    print(
        f"  Depth map     : {rd.shape[1]}x{rd.shape[0]}  "
        f"|  valid after corruption: {n_valid:,} ({n_valid/n_total*100:.1f}%)"
    )
    print(f"  Outliers      : {n_out:,} ({n_out/n_valid*100:.1f}% of valid)")
    print(f"  Clean pixels  : {n_clean:,}")

    if result.sigma_map is not None:
        valid_mask = rd > 0
        sv = result.sigma_map[valid_mask]
        print(f"  σ_hat median  : {np.median(sv):.4f} m  (injected: {NOISE_STD:.4f} m)")

    print(f"\n  Point cloud   : {result.point_cloud.shape[0]:,} points")

    if result.mesh is not None:
        nv = len(result.mesh.vertices)
        nf = len(result.mesh.triangles)
        print(f"  Mesh          : {nv:,} verts, {nf:,} faces")

    if result.mesh_simplified is not None:
        nv2 = len(result.mesh_simplified.vertices)
        nf2 = len(result.mesh_simplified.triangles)
        print(f"  Mesh (simple) : {nv2:,} verts, {nf2:,} faces")

    if result.embedding is not None:
        print(
            f"  Embedding     : {result.embedding.shape[0]}-d  "
            f"(L2={np.linalg.norm(result.embedding):.4f})"
        )

    if result.shape_label:
        print(f"\n  Shape class   : {result.shape_label}")
    if result.shape_descriptors:
        d = result.shape_descriptors
        print(f"  Compactness   : {d['compactness']:.4f}")
        print(f"  Aspect ratio  : {d['aspect_ratio']:.2f}")
        print(f"  Hull ratio    : {d['convex_hull_ratio']:.4f}")

    if result.z_est is not None:
        print(f"\n  Z_est         = {result.z_est:.4f} m")
    if result.z_true is not None:
        print(f"  Z_true        = {result.z_true:.4f} m")
    if result.z_est is not None and result.z_true is not None:
        print(f"  Z error       = {abs(result.z_est - result.z_true)*100:.3f} cm")

    print("\n  Timings:")
    for stage, t in result.timings.items():
        print(f"    {stage:12s}: {t*1000:7.0f} ms")

    # ── Render demo HTML ──────────────────────────────────────────────────────
    print(f"\n--- Generating {OUT_HTML} ---")
    render_pipeline_demo(result, output_path=OUT_HTML)


if __name__ == "__main__":
    main()
