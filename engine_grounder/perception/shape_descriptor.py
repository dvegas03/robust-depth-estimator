"""
Geometric shape descriptors and rule-based classification.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import open3d as o3d


class ShapeDescriptor:
    """Extract geometric properties from an Open3D ``TriangleMesh``."""

    @staticmethod
    def describe(mesh: o3d.geometry.TriangleMesh) -> Dict[str, float]:
        if len(mesh.vertices) < 4:
            return {
                "volume": float("nan"),
                "surface_area": 0.0,
                "compactness": float("nan"),
                "aspect_ratio": float("nan"),
                "convex_hull_ratio": float("nan"),
                "n_vertices": len(mesh.vertices),
                "n_faces": len(mesh.triangles),
            }

        mesh.compute_vertex_normals()

        if not mesh.is_watertight():
            vol = float("nan")
        else:
            vol = mesh.get_volume()

        sa = mesh.get_surface_area()

        if sa > 0 and not math.isnan(vol):
            compactness = (36.0 * math.pi * vol ** 2) / (sa ** 3)
        else:
            compactness = float("nan")

        bbox = mesh.get_axis_aligned_bounding_box()
        extent = np.asarray(bbox.get_extent())
        sorted_ext = np.sort(extent)[::-1]
        aspect_ratio = sorted_ext[0] / sorted_ext[-1] if sorted_ext[-1] > 0 else float("inf")

        # ── EMERGENCY PIVOT: Disable compute_convex_hull() ──
        # Qhull will segfault if the remaining points are coplanar.
        convex_hull_ratio = float("nan")

        return {
            "volume": vol,
            "surface_area": sa,
            "compactness": compactness,
            "aspect_ratio": aspect_ratio,
            "convex_hull_ratio": convex_hull_ratio,
            "n_vertices": len(mesh.vertices),
            "n_faces": len(mesh.triangles),
        }

    @staticmethod
    def classify(desc: Dict[str, float]) -> str:
        c = desc.get("compactness", float("nan"))
        ar = desc.get("aspect_ratio", float("nan"))

        if math.isnan(c):
            return "unknown (non-watertight)"

        if c > 0.85:
            return "sphere-like"

        if 0.5 < c <= 0.85 and ar < 1.5:
            return "box-like / compact"

        if ar > 3.0:
            return "elongated"

        if c < 0.3 and ar < 2.0:
            return "organic / complex surface (animal, plant, etc.)"

        if ar > 2.0:
            return "elongated"

        return "irregular / mixed geometry"

    @staticmethod
    def to_text(desc: Dict[str, float], label: str) -> str:
        lines = [f"Shape classification: {label}"]
        lines.append(f"  Volume           : {desc['volume']:.6f} m^3")
        lines.append(f"  Surface area     : {desc['surface_area']:.4f} m^2")
        lines.append(f"  Compactness      : {desc['compactness']:.4f}")
        lines.append(f"  Aspect ratio     : {desc['aspect_ratio']:.2f}")
        lines.append(f"  Convex-hull ratio: {desc['convex_hull_ratio']:.4f}")
        lines.append(f"  Mesh complexity  : {desc['n_vertices']:,} verts, "
                     f"{desc['n_faces']:,} faces")
        return "\n".join(lines)
