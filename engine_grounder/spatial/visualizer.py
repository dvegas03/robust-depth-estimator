"""
Interactive Plotly 3-D pipeline demo — 6 panels.

Row 1: BEFORE (raw corrupted 3-D cloud) | AFTER (filtered cloud + Z_est)
Row 2: SMOOTHED MESH (Poisson recon)    | EMBEDDING VIZ (PointNet PCA -> RGB)
Row 3: ADAPTIVE sigma MAP (2-D heatmap) | SHAPE PROFILE (table + classification)
"""

from __future__ import annotations

import math
import re
from typing import Optional, Tuple, Dict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_C_VOID     = "#e74c3c"
_C_OUTLIER  = "#e67e22"
_C_ANCHOR   = "#ff2d55"
_C_GTPLANE  = "#2ecc71"
_C_ESTPLANE = "#e74c3c"
_MAX_PTS    = 50_000
_BG         = "#0d0d0d"


def _sub(mask, n, rng):
    idx = np.flatnonzero(mask)
    return idx if idx.size <= n else rng.choice(idx, n, replace=False)


def _bp(depth, fx, fy, cx, cy):
    h, w = depth.shape
    U, V = np.meshgrid(np.arange(w, dtype=np.float32),
                        np.arange(h, dtype=np.float32))
    Z = depth.astype(np.float32).ravel()
    X = (U.ravel() - cx) * Z / fx
    Y = (V.ravel() - cy) * Z / fy
    return X, Y, Z


def _plane(xc, yc, hw, hh, z, color, op, name):
    xs = [xc - hw, xc + hw, xc + hw, xc - hw]
    ys = [yc - hh, yc - hh, yc + hh, yc + hh]
    return go.Mesh3d(x=xs, y=ys, z=[z] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
                     color=color, opacity=op, name=name,
                     showlegend=bool(name), hoverinfo="skip")


_SCENE = dict(
    xaxis=dict(title="X (m)", backgroundcolor="#111", gridcolor="#333",
               showbackground=True, zerolinecolor="#555"),
    yaxis=dict(title="Z (m)", backgroundcolor="#111", gridcolor="#333",
               showbackground=True, zerolinecolor="#555"),
    zaxis=dict(title="Y (m)", backgroundcolor="#111", gridcolor="#333",
               showbackground=True, zerolinecolor="#555",
               autorange="reversed"),
    aspectmode="cube",
    camera=dict(eye=dict(x=-1.2, y=-2.2, z=0.6)),
)


# ══════════════════════════════════════════════════════════════════════════════
# Main demo renderer
# ══════════════════════════════════════════════════════════════════════════════

def render_pipeline_demo(
    result,
    output_path: str = "pipeline_demo.html",
    title: str = "Robust Depth Estimator — Full Pipeline Demo",
):
    """
    Build a 6-panel Plotly HTML from a ``PipelineResult`` dataclass.

    Panels:
        (1,1) Raw corrupted 3-D cloud   (1,2) Filtered cloud + Z_est/GT
        (2,1) Smoothed mesh (Mesh3d)     (2,2) PointNet embedding PCA → RGB
        (3,1) Adaptive σ heatmap         (3,2) Shape profile table
    """
    rng = np.random.default_rng(0)
    hover = "X=%{x:.3f}  Z=%{y:.3f}  Y=%{z:.3f}m"

    # Unpack
    raw   = result.raw_depth
    clean = result.clean_depth
    omask = result.outlier_mask
    smap  = result.sigma_map
    intr  = result.intrinsics
    z_est = result.z_est
    z_true = result.z_true
    mesh  = result.mesh_simplified or result.mesh
    enc_pts = result.encoder_points
    pp_feat = result.per_point_features
    emb     = result.embedding
    desc    = result.shape_descriptors
    label   = result.shape_label
    timings = result.timings

    has_intr = intr is not None
    fx, fy, cx, cy = intr if has_intr else (1, 1, 0, 0)

    # Back-project raw
    X, Y, Z = _bp(raw, fx, fy, cx, cy)
    valid   = np.isfinite(Z) & (Z > 0)
    void    = ~valid
    out_f   = omask.ravel()
    inlier  = valid & ~out_f

    n_total  = raw.size
    n_void   = int(void.sum())
    n_valid  = int(valid.sum())
    n_out    = int((valid & out_f).sum())
    n_inlier = int(inlier.sum())

    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "xy"},    {"type": "domain"}],
        ],
        subplot_titles=[
            "1. Raw Corrupted Point Cloud",
            "2. Filtered (Adaptive Bilateral)",
            "3. Reconstructed Mesh (Poisson)",
            "4. PointNet Embedding (PCA → RGB)",
            "5. Adaptive σ(Z) Map",
            "6. Shape Profile",
        ],
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
    )

    # ── Panel 1: Raw corrupted ────────────────────────────────────────────────
    vi = _sub(void, _MAX_PTS // 6, rng)
    if vi.size:
        fig.add_trace(go.Scatter3d(
            x=X[vi], y=Y[vi], z=np.zeros(vi.size), mode="markers",
            marker=dict(size=1, color=_C_VOID, opacity=0.2),
            name="Void", hovertemplate=hover,
        ), row=1, col=1)

    oi = _sub(valid & out_f, _MAX_PTS // 4, rng)
    if oi.size:
        fig.add_trace(go.Scatter3d(
            x=X[oi], y=Y[oi], z=Z[oi], mode="markers",
            marker=dict(size=1.5, color=_C_OUTLIER, opacity=0.8),
            name="Outlier", hovertemplate=hover,
        ), row=1, col=1)

    ii = _sub(inlier, _MAX_PTS, rng)
    fig.add_trace(go.Scatter3d(
        x=X[ii], y=Y[ii], z=Z[ii], mode="markers",
        marker=dict(size=1, color=Z[ii], colorscale="Viridis", opacity=0.5),
        name="Valid", hovertemplate=hover,
    ), row=1, col=1)

    # ── Panel 2: Filtered + Z_est ─────────────────────────────────────────────
    ii2 = _sub(inlier, _MAX_PTS, rng)
    fig.add_trace(go.Scatter3d(
        x=X[ii2], y=Y[ii2], z=Z[ii2], mode="markers",
        marker=dict(size=1.2, color=Z[ii2], colorscale="Plasma", opacity=0.7,
                    colorbar=dict(title="Z (m)", thickness=6, x=0.48, len=0.28, y=0.88)),
        name="After filter", hovertemplate=hover,
    ), row=1, col=2)

    if z_est is not None and inlier.any():
        xc = float(np.mean(X[inlier]))
        yc = float(np.mean(Y[inlier]))
        hw = float(np.ptp(X[inlier])) / 2
        hh = float(np.ptp(Y[inlier])) / 2
        fig.add_trace(_plane(xc, yc, hw, hh, z_est, _C_ESTPLANE, 0.14,
                             f"Z_est={z_est:.4f}m"), row=1, col=2)
        fig.add_trace(go.Scatter3d(
            x=[xc], y=[yc], z=[z_est], mode="markers+text",
            marker=dict(size=8, color=_C_ANCHOR, symbol="diamond",
                        line=dict(width=2, color="white")),
            text=[f"  Z_est={z_est:.4f}m"],
            textfont=dict(color="white", size=10),
            textposition="middle right",
            name=f"Z_est={z_est:.4f}m",
        ), row=1, col=2)
    if z_true is not None and inlier.any():
        xc2 = float(np.mean(X[inlier]))
        yc2 = float(np.mean(Y[inlier]))
        hw2 = float(np.ptp(X[inlier])) / 2
        hh2 = float(np.ptp(Y[inlier])) / 2
        fig.add_trace(_plane(xc2, yc2, hw2, hh2, z_true, _C_GTPLANE, 0.12,
                             f"GT={z_true:.4f}m"), row=1, col=2)

    # ── Panel 3: Smoothed mesh ────────────────────────────────────────────────
    if mesh is not None:
        verts = np.asarray(mesh.vertices)
        tris  = np.asarray(mesh.triangles)
        # Sub-sample if mesh is huge
        max_faces = 30_000
        if len(tris) > max_faces:
            from engine_grounder.geometry.mesh_builder import MeshBuilder
            mesh_viz = MeshBuilder.simplify(mesh, target_faces=max_faces)
            verts = np.asarray(mesh_viz.vertices)
            tris  = np.asarray(mesh_viz.triangles)

        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 2], z=verts[:, 1],
            i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
            intensity=verts[:, 2],
            colorscale="Viridis",
            colorbar=dict(title="Z (m)", thickness=6, x=0.48, len=0.28, y=0.5),
            opacity=0.85,
            name="Poisson mesh",
            hovertemplate="X=%{x:.3f}  Z=%{y:.3f}  Y=%{z:.3f}m<extra></extra>",
        ), row=2, col=1)

    # ── Panel 4: PointNet embedding PCA → RGB ─────────────────────────────────
    if enc_pts is not None and pp_feat is not None:
        import warnings
        from sklearn.decomposition import PCA

        # Suppress matrix factorization warnings for destroyed point clouds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pca = PCA(n_components=3)
            feat3 = pca.fit_transform(pp_feat)

        # Normalise to [0, 1] for RGB
        fmin = feat3.min(axis=0)
        fmax = feat3.max(axis=0)
        frange = fmax - fmin
        frange[frange == 0] = 1.0
        feat_rgb = (feat3 - fmin) / frange
        colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                  for r, g, b in feat_rgb]

        # Sub-sample for performance
        n_show = min(len(enc_pts), _MAX_PTS)
        stride = max(1, len(enc_pts) // n_show)
        idx = np.arange(0, len(enc_pts), stride)

        fig.add_trace(go.Scatter3d(
            x=enc_pts[idx, 0], y=enc_pts[idx, 2], z=enc_pts[idx, 1],
            mode="markers",
            marker=dict(size=2, color=[colors[i] for i in idx], opacity=0.85),
            name="PointNet features",
            hovertemplate="X=%{x:.3f}  Z=%{y:.3f}  Y=%{z:.3f}<extra></extra>",
        ), row=2, col=2)

    # ── Panel 5: Adaptive σ heatmap ───────────────────────────────────────────
    if smap is not None:
        h_img, w_img = raw.shape
        disp = smap.copy()
        disp[~valid.reshape(h_img, w_img)] = np.nan
        sigma_vals = smap[valid.reshape(h_img, w_img)]
        vmin = float(np.percentile(sigma_vals, 2)) if sigma_vals.size else 0
        vmax = float(np.percentile(sigma_vals, 98)) if sigma_vals.size else 1

        fig.add_trace(go.Heatmap(
            z=np.flipud(disp), colorscale="RdYlGn_r",
            zmin=vmin, zmax=vmax,
            colorbar=dict(title="σ (m)", thickness=6, x=0.48, len=0.22, y=0.12,
                          tickformat=".3f"),
            hoverongaps=False,
            hovertemplate="σ=%{z:.4f}m<extra></extra>",
            name="σ(Z) map",
        ), row=3, col=1)

        fig.update_xaxes(title_text="u (px)", showgrid=False, color="white",
                         row=3, col=1)
        fig.update_yaxes(title_text="v (px)", showgrid=False, color="white",
                         row=3, col=1)

    # ── Panel 6: Shape profile table ──────────────────────────────────────────
    table_header = ["Property", "Value"]
    rows_k = []
    rows_v = []

    if desc:
        rows_k += ["Volume", "Surface area", "Compactness", "Aspect ratio",
                    "Convex-hull ratio", "Vertices", "Faces"]
        rows_v += [
            f"{desc['volume']:.6f} m³",
            f"{desc['surface_area']:.4f} m²",
            f"{desc['compactness']:.4f}",
            f"{desc['aspect_ratio']:.2f}",
            f"{desc['convex_hull_ratio']:.4f}",
            f"{desc['n_vertices']:,}",
            f"{desc['n_faces']:,}",
        ]

    if label:
        rows_k.append("Classification")
        rows_v.append(f"<b>{label}</b>")

    if emb is not None:
        rows_k.append("Embedding dim")
        rows_v.append(str(emb.shape[0]))
        rows_k.append("Embedding L2 norm")
        rows_v.append(f"{np.linalg.norm(emb):.4f}")

    if z_est is not None:
        rows_k.append("Z_est")
        rows_v.append(f"{z_est:.4f} m")
    if z_true is not None:
        rows_k.append("Z_true (median)")
        rows_v.append(f"{z_true:.4f} m")
    if z_est is not None and z_true is not None:
        rows_k.append("Z error")
        rows_v.append(f"{abs(z_est - z_true)*100:.3f} cm")

    for stage, t in timings.items():
        rows_k.append(f"Time: {stage}")
        rows_v.append(f"{t*1000:.0f} ms")

    fig.add_trace(go.Table(
        header=dict(
            values=table_header,
            fill_color="#1a1a1a",
            font=dict(color="white", size=11),
            align="left",
        ),
        cells=dict(
            values=[rows_k, rows_v],
            fill_color="#111",
            font=dict(color="white", size=10),
            align="left",
            height=22,
        ),
    ), row=3, col=2)

    # ── Layout ────────────────────────────────────────────────────────────────
    void_pct = n_void / n_total * 100 if n_total else 0
    out_pct  = n_out / n_valid * 100 if n_valid else 0

    scene_kw = {}
    for key in ("scene", "scene2", "scene3", "scene4"):
        scene_kw[key] = _SCENE

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="white"), x=0.5),
        **scene_kw,
        legend=dict(x=0.5, y=-0.02, xanchor="center", orientation="h",
                    font=dict(size=8, color="white"),
                    bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(l=0, r=10, t=80, b=10),
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        height=1400,
    )

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color="white")

    fig.write_html(output_path, auto_open=True)
    print(f"-> Demo saved -> {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Legacy single-panel helper (kept for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def render_before_after(
    raw_depth, stable_z, intrinsics, outlier_mask,
    sigma_map=None, true_depth=None,
    output_path="depth_validation.html",
    title="RobustDepthEstimator",
):
    """Thin wrapper: builds a minimal PipelineResult and calls the demo renderer."""
    from dataclasses import dataclass

    class _Mini:
        pass

    r = _Mini()
    r.raw_depth = raw_depth
    r.clean_depth = raw_depth.copy()
    r.clean_depth[outlier_mask] = 0.0
    r.outlier_mask = outlier_mask
    r.sigma_map = sigma_map
    r.thresh_map = None
    r.intrinsics = intrinsics
    r.rgb = None
    r.z_est = stable_z
    r.z_true = true_depth
    r.mesh = None
    r.mesh_simplified = None
    r.embedding = None
    r.per_point_features = None
    r.encoder_points = None
    r.shape_descriptors = None
    r.shape_label = None
    r.shape_text = None
    r.timings = {}
    r.point_cloud = None

    render_pipeline_demo(r, output_path=output_path, title=title)


class PointCloudVisualizer:
    render_before_after = staticmethod(render_before_after)
    render_pipeline_demo = staticmethod(render_pipeline_demo)
