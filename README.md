# Robust Depth Estimator

**Semantic–Geometric Decoupling Framework**

[![CI](https://github.com/dvegas03/robust-depth-estimator/actions/workflows/ci.yml/badge.svg)](https://github.com/dvegas03/robust-depth-estimator/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

Robust Depth Estimator is a production-grade open-source framework implementing **Semantic–Geometric Decoupling**: sensor noise and geometric structure are estimated jointly and adaptively, so downstream spatial reasoning is not contaminated by depth sensor artefacts.

## Architecture

```
RGB-D Sensor
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│  Stage 1 — Adaptive Bilateral-Residual Depth Filter            │
│  • Depth-binned σ(Z) model — no manual noise_sigma required    │
│  • Void masking → median pre-filter → bilateral residual       │
│  • Morphological closing + OpenCV Telea inpainting             │
└──────────────────────────┬─────────────────────────────────────┘
                           │  clean depth map
                           ▼
┌────────────────────────────────────────────────────────────────┐
│  Stage 2 — Pinhole Back-Projection  (Projector)                │
│  • Vectorised: X=(u−cx)·Z/fx,  Y=(v−cy)·Z/fy                 │
└──────────────────────────┬─────────────────────────────────────┘
                           │  (N,3) point cloud
                           ▼
┌────────────────────────────────────────────────────────────────┐
│  Stage 3 — Mesh Reconstruction  (MeshBuilder)                  │
│  • Open3D alpha-shape (stable on Apple Silicon)                │
│  • Quadric decimation → compact mesh                           │
└──────────────────────────┬─────────────────────────────────────┘
                           │  TriangleMesh
                           ▼
┌────────────────────────────────────────────────────────────────┐
│  Stage 4 — PointNet Shape Encoding  (ShapeEncoder)             │
│  • Shared-MLP 3→64→128→1024 + max-pool + FC 1024→512→256      │
│  • 256-d global embedding  +  (N,1024) per-point features      │
└──────────────────────────┬─────────────────────────────────────┘
                           │  embedding vector
                           ▼
┌────────────────────────────────────────────────────────────────┐
│  Stage 5 — Geometric Shape Description  (ShapeDescriptor)      │
│  • Volume, surface area, compactness, aspect ratio             │
│  • Rule-based classifier: sphere / box / elongated / organic   │
└──────────────────────────┬─────────────────────────────────────┘
                           │  shape text + embedding
                           ▼
┌────────────────────────────────────────────────────────────────┐
│  Stage 6 — VLM Context Fusion  (VLMAgent)                      │
│  • Stores spatial context for next Qwen-VL query               │
└────────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
pip install -e ".[viz]"

# Generate the Stanford Bunny depth map (run once)
OMP_NUM_THREADS=1 python examples/prepare_bunny.py

# Run the full pipeline + interactive 6-panel HTML demo
OMP_NUM_THREADS=1 python examples/main.py
```

```python
from engine_grounder import Pipeline, BunnyStream

result = Pipeline(BunnyStream()).run()
print(f"Z_est = {result.z_est:.4f} m  |  shape = {result.shape_label}")
print(f"Embedding dim: {result.embedding.shape[0]}")
```

## Installation

```bash
# Core pipeline only (no Plotly)
pip install engine-grounder

# With interactive visualizations
pip install "engine-grounder[viz]"

# Full install including dev tools
pip install "engine-grounder[all]"
```

**Apple Silicon note:** To avoid an OpenMP conflict between PyTorch and Open3D, always prepend:
```bash
OMP_NUM_THREADS=1 python ...
```

## Key Design Decisions

### Adaptive σ(Z) Noise Model
I implemented a depth-binned half-normal estimator that recovers the true sensor noise σ(Z) directly from bilateral residuals — no manual `noise_sigma` parameter is required. This is mathematically grounded in the half-normal median identity:

```
σ̂_bin = Q50(|residuals|_bin) / 0.6745
```

The result: the filter adapts to each sensor's actual noise profile and generalises to new hardware without re-tuning.

### Semantic-Geometric Decoupling
Rather than asking a VLM to reason about raw noisy depth, I route the geometry through a deterministic cleaning and encoding stage first. The VLM receives a 256-d PointNet embedding and a natural-language shape profile alongside the image — grounding its spatial responses in physically meaningful geometry.

### Apple Silicon Stability
Open3D's Qhull convex hull segfaults on coplanar point sets generated on Apple Silicon. I replaced it with alpha-shape reconstruction, which is numerically stable on M-series chips.

## Hardware Benchmarks

Measured on the Stanford Bunny (480×640 depth map, 55% void injection, σ = 0.015 m):

| Stage         | Apple M4 Pro | RTX 4070 (Ubuntu) |
|---------------|:------------:|:-----------------:|
| Depth filter  | ~18 ms       | ~9 ms             |
| Back-project  | ~2 ms        | ~1 ms             |
| Mesh build    | ~120 ms      | ~60 ms            |
| PointNet enc  | ~45 ms       | ~8 ms             |
| Shape describe| ~5 ms        | ~3 ms             |
| **Total**     | **~190 ms**  | **~81 ms**        |

Z estimation error: **< 0.5 cm** across void ratios 10–85% and noise levels 0.005–0.05 m.

## Project Structure

```
robust-depth-estimator/
├── engine_grounder/        # pip-installable Python package
│   ├── pipeline.py         # 6-stage orchestrator
│   ├── geometry/           # depth_filter.py, mesh_builder.py
│   ├── spatial/            # projector.py, visualizer.py
│   ├── perception/         # shape_encoder.py, shape_descriptor.py, vlm_agent.py
│   ├── streams/            # sensor_stream.py, mock_stream.py, quest_stream.py
│   └── utils/              # synthetic_data.py
├── tests/                  # 9 test modules, 100+ test cases
├── benchmarks/             # Comprehensive parameter sweep + HTML report
├── examples/               # main.py, render_split_viz.py, prepare_bunny.py
├── tools/                  # generate_poster_qr.py
├── assets/                 # github_logo.png (README images)
├── data/                   # depth maps + intrinsics (gitignored)
├── pyproject.toml          # Modern PEP 517 packaging
├── .pre-commit-config.yaml # ruff lint + format on commit
└── .github/workflows/      # CI: lint + test (Ubuntu + macOS matrix)
```

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers void masking, adaptive σ estimation, bilateral outlier detection, IQR filter, morphological closing, inpainting, bilateral smoothing, full restore pipeline, Z_est extraction, Monte Carlo accuracy, mesh reconstruction, point cloud projector math, shape encoding, shape descriptors, and synthetic data generation.

## Citation (I3D 2026)

If you use Robust Depth Estimator in your research, please cite:

```bibtex
@inproceedings{kharitonenkov2026robust,
  author  = {Kharitonenkov, Danila and Chheang, Vuthea},
  title   = {{Robust Depth Estimator}: Semantic--Geometric Decoupling Framework},
  booktitle = {Proceedings of I3D},
  year    = {2026},
  url     = {https://github.com/dvegas03/robust-depth-estimator},
}
```

Or use the **"Cite this repository"** button on GitHub (powered by `CITATION.cff`).

## License

MIT — see [LICENSE](LICENSE).
