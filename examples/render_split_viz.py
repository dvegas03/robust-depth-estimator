"""
Geometric Decoupling Cinematic Visualizer
======================================
Stanford Bunny — Cinematic "ground up" materialization and degradation.
Coordinates are injected as JSON; a custom JS engine streams points into Plotly.

Run from engine_grounder/:
    OMP_NUM_THREADS=1 .venv/bin/python render_split_viz.py
"""

from __future__ import annotations

import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import plotly.graph_objects as go

from engine_grounder.streams.mock_stream import BunnyStream
from engine_grounder.pipeline import Pipeline

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
OUT_HTML = os.path.join(_REPO_ROOT, "split_viz.html")
VOID_RATIO = 0.55
NOISE_STD = 0.015
OUTLIER_K = 3.0

# ── Run pipeline ──────────────────────────────────────────────────────────────
print("Running pipeline on Stanford Bunny …")
rng_corrupt = np.random.default_rng(7)


def inject(depth: np.ndarray) -> np.ndarray:
    out = depth.copy()
    vi = np.flatnonzero(out > 0)
    n = int(len(vi) * VOID_RATIO)
    out.flat[rng_corrupt.choice(vi, n, replace=False)] = 0.0
    still = out > 0
    out[still] += rng_corrupt.normal(0, NOISE_STD, still.sum()).astype(np.float32)
    np.clip(out, 0, None, out=out)
    return out


stream = BunnyStream(
    depth_path=os.path.join(DATA_DIR, "bunny_depth.npy"),
    intrinsics_path=os.path.join(DATA_DIR, "bunny_intrinsics.npy"),
)  
result = Pipeline(stream, outlier_k=OUTLIER_K, tau=0.10).run(corrupt_fn=inject)
print("  Pipeline done. Preparing cinematic data…")

# ── Unpack ────────────────────────────────────────────────────────────────────
raw = result.raw_depth
omask = result.outlier_mask
smap = result.sigma_map
fx, fy, cx, cy = result.intrinsics
true_depth = np.load(os.path.join(DATA_DIR, "bunny_depth.npy")).astype(np.float32)  

rng = np.random.default_rng(7)


def _bp(depth, fx, fy, cx, cy):
    h, w = depth.shape
    U, V = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    Z = depth.ravel().astype(np.float32)
    X = (U.ravel() - cx) * Z / fx
    Y = (V.ravel() - cy) * Z / fy
    return X, Y, Z


Xr, Yr, Zr = _bp(raw, fx, fy, cx, cy)
Xt, Yt, Zt = _bp(true_depth, fx, fy, cx, cy)

valid_true = Zt > 0
raw_flat = raw.ravel().astype(np.float32)
void_mask = (raw_flat == 0) & valid_true
stay_mask = (raw_flat > 0) & valid_true
outlier_mask = stay_mask & omask.ravel().astype(bool)


def get_dance_subset(mask, max_pts):
    idx = np.flatnonzero(mask)
    if idx.size > max_pts:
        idx = rng.choice(idx, max_pts, replace=False)
    fuzzy_y = Yt[idx] + rng.uniform(-0.06, 0.06, len(idx))
    sort_order = np.argsort(-fuzzy_y)
    return idx[sort_order]


idx_void = get_dance_subset(void_mask, 14_000)
idx_stay = get_dance_subset(stay_mask, 14_000)
idx_out = get_dance_subset(outlier_mask, 3500)

center_z = float(np.median(Zr[stay_mask]))
spike_z = center_z + (Zr[idx_out] - center_z) * rng.uniform(1.2, 2.8, len(idx_out))

# Plotly scene: x=X, y=Z (depth), z=Y (vertical)
data_payload = json.dumps(
    {
        "t0_x": Xt[idx_void].tolist(),
        "t0_y": Zt[idx_void].tolist(),
        "t0_z": Yt[idx_void].tolist(),
        "t1_x": Xt[idx_stay].tolist(),
        "t1_y": Zt[idx_stay].tolist(),
        "t1_z": Yt[idx_stay].tolist(),
        "t2_x": Xr[idx_stay].tolist(),
        "t2_y": Zr[idx_stay].tolist(),
        "t2_z": Yr[idx_stay].tolist(),
        "t2_c": Zr[idx_stay].tolist(),
        "t3_x": Xr[idx_out].tolist(),
        "t3_y": spike_z.tolist(),
        "t3_z": Yr[idx_out].tolist(),
    }
)

# Metrics for optional HUD
n_valid = int(stay_mask.sum())
n_void_pts = int(void_mask.sum())
n_out_pts = int(outlier_mask.sum())
z_err_cm = (
    abs(result.z_est - result.z_true) * 100
    if (result.z_est and result.z_true)
    else 0.0
)
mesh_f = len(result.mesh.triangles) if result.mesh else 0

# ── Figure (empty scatters; JS fills coordinates) ─────────────────────────────
fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode="markers",
        marker=dict(size=1.8, color="#00e5ff", opacity=0.7),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode="markers",
        marker=dict(size=1.8, color="#00e5ff", opacity=0.7),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode="markers",
        marker=dict(
            size=2.2,
            color=[],
            colorscale=[
                [0.0, "#b30000"],
                [0.5, "#e03000"],
                [1.0, "#e87000"],
            ],
            cmin=float(Zr[stay_mask].min()),
            cmax=float(Zr[stay_mask].max()),
            opacity=0.9,
        ),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode="markers",
        marker=dict(size=3.8, color="#cc1100", symbol="diamond", opacity=1.0),
        hoverinfo="skip",
    )
)

mesh_trace_idx: int | None = None
wireframe_indices: list[int] = []

mesh = result.mesh_simplified or result.mesh
if mesh is not None and len(mesh.triangles) > 0:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    tris = np.asarray(mesh.triangles)
    depth_per_vert = verts[:, 2]

    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 2],
            z=verts[:, 1],
            i=tris[:, 0],
            j=tris[:, 1],
            k=tris[:, 2],
            intensity=depth_per_vert,
            colorscale=[
                [0.0, "#003850"],
                [0.4, "#005f7a"],
                [0.7, "#007f8a"],
                [1.0, "#009a8e"],
            ],
            cmin=float(depth_per_vert.min()),
            cmax=float(depth_per_vert.max()),
            opacity=0.95,
            flatshading=True,
            lighting=dict(
                ambient=0.70,
                diffuse=0.95,
                specular=0.45,
                roughness=0.25,
                fresnel=0.15,
            ),
            lightposition=dict(x=-1, y=-2, z=2),
            visible=False,
            hoverinfo="skip",
        )
    )
    mesh_trace_idx = len(fig.data) - 1

    wire_step = max(1, len(tris) // 1500)
    for tri in tris[::wire_step]:
        vs = verts[tri]
        fig.add_trace(
            go.Scatter3d(
                x=[vs[0, 0], vs[1, 0], vs[2, 0], vs[0, 0]],
                y=[vs[0, 2], vs[1, 2], vs[2, 2], vs[0, 2]],
                z=[vs[0, 1], vs[1, 1], vs[2, 1], vs[0, 1]],
                mode="lines",
                line=dict(color="rgba(0,50,80,0.15)", width=1),
                visible=False,
                hoverinfo="skip",
            )
        )
        wireframe_indices.append(len(fig.data) - 1)

has_mesh = mesh_trace_idx is not None
mesh_idx_json = json.dumps(mesh_trace_idx)
wire_idx_json = json.dumps(wireframe_indices)

_BG = "#ffffff"
fig.update_layout(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    margin=dict(l=0, r=0, t=0, b=0),
    scene=dict(
        xaxis=dict(
            title="X (m)",
            backgroundcolor="#f0f4f8",
            gridcolor="#ccd6e0",
            zerolinecolor="#99aabb",
        ),
        yaxis=dict(
            title="Depth (m)",
            backgroundcolor="#f0f4f8",
            gridcolor="#ccd6e0",
            zerolinecolor="#99aabb",
        ),
        zaxis=dict(
            title="Y (m)",
            backgroundcolor="#f0f4f8",
            gridcolor="#ccd6e0",
            zerolinecolor="#99aabb",
            autorange="reversed",
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=0.0, y=-2.8, z=0.25), up=dict(x=0, y=0, z=1)),
        bgcolor=_BG,
    ),
    showlegend=False,
)

plot_div = fig.to_html(
    full_html=False,
    include_plotlyjs="cdn",
    config=dict(displayModeBar=False, displaylogo=False, scrollZoom=True),
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Cinematic Geometric Decoupling</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #ffffff; font-family: 'Courier New', monospace; overflow: hidden; width: 100vw; height: 100vh; }}
  #plot-container {{ position: absolute; inset: 0; }}
  #plot-container > div {{ width: 100% !important; height: 100% !important; }}
  #hud {{ position: fixed; inset: 0; pointer-events: none; z-index: 500; }}

  #play-btn {{
    position: absolute; bottom: 100px; left: 50%; transform: translateX(-50%);
    padding: 14px 34px; background: #005f50; color: white; border: 1.5px solid #009a8e;
    border-radius: 4px; font-size: 15px; font-weight: bold; cursor: pointer;
    box-shadow: 0 4px 20px rgba(0,95,80,0.4); z-index: 1000; letter-spacing: 3px;
    pointer-events: auto; transition: all 0.3s ease; font-family: inherit;
  }}
  #play-btn:hover {{ background: #007f6a; transform: translateX(-50%) scale(1.03); }}

  #top-bar {{ position: absolute; top: 0; left: 0; right: 0; height: 70px; display: flex; align-items: center; justify-content: space-between; padding: 0 34px; background: rgba(255,255,255,0.96); border-bottom: 1.5px solid #dde3ea; box-shadow: 0 1px 8px rgba(0,0,0,0.07); }}
  .bar-left {{ color: #006655; font-size: 17px; letter-spacing: 3.5px; font-weight: 600; transition: color 0.5s; }}
  .bar-mid {{ color: #8899aa; font-size: 14px; letter-spacing: 4px; }}
  .bar-right {{ color: #006655; font-size: 17px; letter-spacing: 3.5px; font-weight: 600; opacity: 0.3; transition: opacity 1s; }}

  .callout {{ position: absolute; left: 42px; top: 22%; padding: 18px 22px 20px; font-size: 14.5px; line-height: 1.8; letter-spacing: 0.8px; border-radius: 4px; box-shadow: 0 2px 16px rgba(0,0,0,0.10); transition: all 0.5s ease; border: 1.5px solid #80c4be; background: rgba(242,252,250,0.97); color: #003d35; width: 340px; opacity: 0; }}
  .callout .title {{ font-size: 13px; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1.5px solid currentColor; }}
  .val {{ font-size: 15px; font-weight: 600; float: right; }}
  .state-warn {{ border-color: #e8b0a0 !important; background: rgba(255,248,245,0.97) !important; color: #b82000 !important; }}
  .state-good {{ border-color: #80c4be !important; background: rgba(242,252,250,0.97) !important; color: #005f50 !important; }}
  @keyframes blink {{ 0%,100%{{ opacity:1 }} 50%{{ opacity:0.2 }} }} .blink {{ animation: blink 1.4s step-start infinite; }}
</style>
</head>
<body>

<div id="plot-container">{plot_div}</div>

<div id="hud">
  <button type="button" id="play-btn">▶ INITIALIZE SENSOR CAPTURE</button>

  <div id="top-bar">
    <div class="bar-left" id="sys-title">SYSTEM: STANDBY</div>
    <div class="bar-mid">ENGINE GROUNDER VIZ</div>
    <div class="bar-right" id="sys-opt">ALGO: ALPHA-SHAPE <span class="blink">| ✓ OPTIMIZED</span></div>
  </div>

  <div class="callout" id="main-callout">
    <div class="title" id="co-title">▶ ACQUIRING GEOMETRY</div>
    STATUS&nbsp;&nbsp;&nbsp;: <span class="val" id="co-status">SCANNING…</span><br>
    POINTS&nbsp;&nbsp;&nbsp;: <span class="val" id="co-pts">0</span><br>
    <div id="co-extra"></div>
  </div>
</div>

<script>
// Pacing: larger increments / shorter delays = snappier sequence (~2.5–3× faster than v1)
var BUILD_PTS_PER_FRAME = 240;
var HOLD_CLEAN_MS = 650;
var VOID_PTS_PER_FRAME = 340;
var PAUSE_BEFORE_GLITCH_MS = 350;
var SPIKE_PTS_PER_FRAME = 150;
var HOLD_MESSY_MS = 900;
// Orbit: slower during point-cloud phases; faster only after mesh reconstruction (phase >= 6)
var ORBIT_RAD_PER_FRAME = 0.0045;
var ORBIT_RAD_PER_FRAME_MESH = 0.018;
var WIRE_DELAY_MS = 120;

var rawData = {data_payload};
var HAS_MESH = {str(has_mesh).lower()};
var MESH_TRACE_IDX = {mesh_idx_json};
var WIREFRAME_INDICES = {wire_idx_json};
var Z_ERR_CM = {z_err_cm:.3f};

(function resize() {{
  var el = document.querySelector('#plot-container > div');
  if (!el) {{ setTimeout(resize, 100); return; }}
  el.style.width = window.innerWidth + 'px';
  el.style.height = window.innerHeight + 'px';
  window.addEventListener('resize', function() {{
    el.style.width = window.innerWidth + 'px';
    el.style.height = window.innerHeight + 'px';
    if (window.Plotly) Plotly.relayout(el, {{ width: window.innerWidth, height: window.innerHeight }});
  }});
}})();

function getGd() {{
  return document.querySelector('#plot-container .js-plotly-plot') || document.querySelector('#plot-container > div');
}}

var isPlaying = false;
var theta = -Math.PI / 2;
var radius = 2.8;
var phase = 0;
var f_t01 = 0;
var f_t0_deg = rawData.t0_x.length;
var f_t3 = 0;
var meshRevealScheduled = false;

var max_t0 = rawData.t0_x.length;
var max_t1 = rawData.t1_x.length;
var max_t3 = rawData.t3_x.length;

function updateTrace(idx, prefix, count, colorKey) {{
  var gd = getGd();
  if (!gd || !window.Plotly) return;
  var update = {{
    x: [rawData[prefix + '_x'].slice(0, count)],
    y: [rawData[prefix + '_y'].slice(0, count)],
    z: [rawData[prefix + '_z'].slice(0, count)]
  }};
  if (colorKey) update['marker.color'] = [rawData[colorKey].slice(0, count)];
  Plotly.restyle(gd, update, [idx]);
}}

var co = document.getElementById('main-callout');
var coTitle = document.getElementById('co-title');
var coStatus = document.getElementById('co-status');
var coPts = document.getElementById('co-pts');
var coExtra = document.getElementById('co-extra');

document.getElementById('play-btn').addEventListener('click', function() {{
  if (isPlaying) return;
  isPlaying = true;
  this.style.opacity = '0';
  this.style.pointerEvents = 'none';
  meshRevealScheduled = false;
  document.getElementById('sys-title').innerText = 'SYSTEM: CAPTURING';
  document.getElementById('sys-title').classList.add('blink');
  co.style.opacity = '1';
  requestAnimationFrame(renderLoop);
}});

function renderLoop() {{
  var el = getGd();
  if (window.Plotly && el) {{
    var orbitStep = phase >= 6 ? ORBIT_RAD_PER_FRAME_MESH : ORBIT_RAD_PER_FRAME;
    theta += orbitStep;
    Plotly.relayout(el, {{
      'scene.camera.eye.x': radius * Math.cos(theta),
      'scene.camera.eye.y': radius * Math.sin(theta),
      'scene.camera.eye.z': 0.25
    }});
  }}

  if (phase === 0) {{
    f_t01 += BUILD_PTS_PER_FRAME;
    if (f_t01 > Math.max(max_t0, max_t1)) f_t01 = Math.max(max_t0, max_t1);
    updateTrace(0, 't0', Math.min(f_t01, max_t0));
    updateTrace(1, 't1', Math.min(f_t01, max_t1));
    coPts.innerText = (Math.min(f_t01, max_t0) + Math.min(f_t01, max_t1)).toLocaleString();
    if (f_t01 >= Math.max(max_t0, max_t1)) {{
      phase = 1;
      setTimeout(function() {{ phase = 2; }}, HOLD_CLEAN_MS);
    }}
  }} else if (phase === 2) {{
    if (f_t0_deg === max_t0) {{
      co.classList.add('state-warn');
      coTitle.innerText = '▶ CORRUPTION DETECTED';
      coStatus.innerText = 'BLACK HOLE EFFECT';
      document.getElementById('sys-title').style.color = '#b82000';
      document.getElementById('sys-title').innerText = 'SYSTEM: DEGRADED';
    }}
    f_t0_deg -= VOID_PTS_PER_FRAME;
    if (f_t0_deg < 0) f_t0_deg = 0;
    updateTrace(0, 't0', f_t0_deg);
    coPts.innerText = (f_t0_deg + max_t1).toLocaleString();
    if (f_t0_deg === 0) {{
      phase = 3;
      setTimeout(function() {{ phase = 4; }}, PAUSE_BEFORE_GLITCH_MS);
    }}
  }} else if (phase === 4) {{
    if (f_t3 === 0) {{
      coStatus.innerText = 'MULTI-PATH NOISE';
      updateTrace(1, 't1', 0);
      updateTrace(2, 't2', max_t1, 't2_c');
    }}
    f_t3 += SPIKE_PTS_PER_FRAME;
    if (f_t3 > max_t3) f_t3 = max_t3;
    updateTrace(3, 't3', f_t3);
    coExtra.innerHTML = 'OUTLIERS&nbsp;: <span class="val" style="color:#b82000">' + f_t3 + ' px</span>';
    if (f_t3 === max_t3) {{
      phase = 5;
      setTimeout(function() {{ phase = 6; }}, HOLD_MESSY_MS);
    }}
  }} else if (phase === 6) {{
    if (!meshRevealScheduled) {{
      meshRevealScheduled = true;
      var gd = getGd();
      if (HAS_MESH && gd && MESH_TRACE_IDX !== null) {{
        Plotly.restyle(gd, {{ visible: false }}, [0, 1, 2, 3]);
        Plotly.restyle(gd, {{ visible: true }}, [MESH_TRACE_IDX]);
        setTimeout(function() {{
          if (WIREFRAME_INDICES.length)
            Plotly.restyle(gd, {{ visible: true }}, WIREFRAME_INDICES);
        }}, WIRE_DELAY_MS);
      }} else {{
        if (gd) Plotly.restyle(gd, {{ visible: false }}, [0, 1, 2, 3]);
        coExtra.innerHTML = '<span class="val">No mesh in this build</span>';
      }}
      co.classList.remove('state-warn');
      co.classList.add('state-good');
      coTitle.innerText = '▶ FILTER APPLIED';
      coStatus.innerText = 'GEOMETRY DECOUPLED';
      coPts.innerText = '—';
      coExtra.innerHTML = 'Z_ERROR&nbsp;&nbsp;: <span class="val" style="color:#005f50">' + Z_ERR_CM.toFixed(3) + ' cm</span>';
      document.getElementById('sys-title').style.color = '#006655';
      document.getElementById('sys-title').innerText = 'SYSTEM: STABLE';
      document.getElementById('sys-title').classList.remove('blink');
      document.getElementById('sys-opt').style.opacity = '1';
    }}
    phase = 7;
  }}

  requestAnimationFrame(renderLoop);
}}
</script>
</body>
</html>
"""

with open(OUT_HTML, "w", encoding="utf-8") as fh:
    fh.write(html)

print(f"\n✓  Cinematic sequence injected → {OUT_HTML}")
print(f"   Click ▶ INITIALIZE SENSOR CAPTURE — ground-up dance, degradation, mesh reveal.")
print(f"   Mesh trace index: {mesh_trace_idx}, wire frames: {len(wireframe_indices)}")
