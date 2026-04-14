"""Tests for ShapeDescriptor — geometric properties + classification."""

import os

import math
import numpy as np
import open3d as o3d
import pytest

from engine_grounder.perception.shape_descriptor import ShapeDescriptor


def _sphere_mesh(radius=1.0, resolution=20):
    return o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)


def _box_mesh(w=1.0, h=1.0, d=1.0):
    return o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=d)


def _cylinder_mesh(radius=0.2, height=3.0, resolution=30):
    return o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height,
                                                       resolution=resolution)


class TestDescribe:
    def test_returns_dict(self):
        d = ShapeDescriptor.describe(_sphere_mesh())
        assert isinstance(d, dict)

    def test_has_all_keys(self):
        d = ShapeDescriptor.describe(_sphere_mesh())
        for k in ("volume", "surface_area", "compactness", "aspect_ratio",
                   "convex_hull_ratio", "n_vertices", "n_faces"):
            assert k in d, f"Missing key: {k}"

    def test_sphere_volume(self):
        r = 1.0
        d = ShapeDescriptor.describe(_sphere_mesh(radius=r, resolution=30))
        expected = (4 / 3) * math.pi * r ** 3
        assert abs(d["volume"] - expected) / expected < 0.1

    def test_sphere_compactness_near_one(self):
        d = ShapeDescriptor.describe(_sphere_mesh(resolution=30))
        assert d["compactness"] > 0.8

    def test_box_compactness_below_sphere(self):
        d_sphere = ShapeDescriptor.describe(_sphere_mesh(resolution=30))
        d_box = ShapeDescriptor.describe(_box_mesh())
        assert d_box["compactness"] < d_sphere["compactness"]

    def test_elongated_aspect_ratio(self):
        d = ShapeDescriptor.describe(_cylinder_mesh(radius=0.2, height=5.0))
        assert d["aspect_ratio"] > 2.0

    def test_cube_aspect_ratio_near_one(self):
        d = ShapeDescriptor.describe(_box_mesh(1, 1, 1))
        assert d["aspect_ratio"] < 1.5

    def test_vertex_face_counts(self):
        mesh = _sphere_mesh()
        d = ShapeDescriptor.describe(mesh)
        assert d["n_vertices"] == len(mesh.vertices)
        assert d["n_faces"] == len(mesh.triangles)


class TestClassify:
    def test_sphere_classified(self):
        d = ShapeDescriptor.describe(_sphere_mesh(resolution=30))
        label = ShapeDescriptor.classify(d)
        assert "sphere" in label.lower()

    def test_elongated_classified(self):
        d = ShapeDescriptor.describe(_cylinder_mesh(radius=0.1, height=5.0))
        label = ShapeDescriptor.classify(d)
        assert "elongated" in label.lower() or "irregular" in label.lower()

    def test_returns_string(self):
        d = ShapeDescriptor.describe(_box_mesh())
        assert isinstance(ShapeDescriptor.classify(d), str)


class TestToText:
    def test_contains_label(self):
        d = ShapeDescriptor.describe(_sphere_mesh())
        label = ShapeDescriptor.classify(d)
        text = ShapeDescriptor.to_text(d, label)
        assert label in text

    def test_contains_volume(self):
        d = ShapeDescriptor.describe(_sphere_mesh())
        text = ShapeDescriptor.to_text(d, "test")
        assert "Volume" in text

    def test_returns_string(self):
        d = ShapeDescriptor.describe(_sphere_mesh())
        assert isinstance(ShapeDescriptor.to_text(d, "x"), str)
