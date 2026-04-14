"""Tests for MeshBuilder — Poisson reconstruction, simplification, sampling."""

import os

import numpy as np
import open3d as o3d
import pytest

from engine_grounder.geometry.mesh_builder import MeshBuilder


def _sphere_cloud_with_normals(n=3000, radius=1.0, seed=0):
    """Sphere cloud with analytically correct outward normals (fast path)."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, n)
    cos_theta = rng.uniform(-1, 1, n)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    pts = np.column_stack([
        radius * sin_theta * np.cos(phi),
        radius * sin_theta * np.sin(phi),
        radius * cos_theta,
    ])
    normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts, normals


class TestFromPointCloud:
    def test_returns_triangle_mesh(self):
        pts, norms = _sphere_cloud_with_normals(n=2000)
        mesh = MeshBuilder().from_point_cloud(pts, normals=norms)
        assert isinstance(mesh, o3d.geometry.TriangleMesh)

    def test_mesh_has_vertices_and_faces(self):
        pts, norms = _sphere_cloud_with_normals(n=2000)
        mesh = MeshBuilder().from_point_cloud(pts, normals=norms)
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0

    def test_custom_normals_accepted(self):
        pts, norms = _sphere_cloud_with_normals()
        mesh = MeshBuilder().from_point_cloud(pts, normals=norms)
        assert len(mesh.vertices) > 0


class TestSimplify:
    def test_reduces_face_count(self):
        pts, norms = _sphere_cloud_with_normals(n=3000)
        mesh = MeshBuilder().from_point_cloud(pts, normals=norms)
        n_before = len(mesh.triangles)
        simplified = MeshBuilder.simplify(mesh, target_faces=500)
        assert len(simplified.triangles) <= n_before
        assert len(simplified.triangles) <= 600

    def test_returns_triangle_mesh(self):
        pts, norms = _sphere_cloud_with_normals(n=2000)
        mesh = MeshBuilder().from_point_cloud(pts, normals=norms)
        s = MeshBuilder.simplify(mesh, target_faces=200)
        assert isinstance(s, o3d.geometry.TriangleMesh)


class TestSamplePoints:
    def test_output_shape(self):
        pts, norms = _sphere_cloud_with_normals(n=2000)
        mesh = MeshBuilder().from_point_cloud(pts, normals=norms)
        sampled = MeshBuilder.sample_points(mesh, n=1024)
        assert sampled.shape == (1024, 3)

    def test_dtype_float64(self):
        pts, norms = _sphere_cloud_with_normals(n=2000)
        mesh = MeshBuilder().from_point_cloud(pts, normals=norms)
        sampled = MeshBuilder.sample_points(mesh, n=100)
        assert sampled.dtype == np.float64

    def test_points_near_original_surface(self):
        pts, norms = _sphere_cloud_with_normals(n=3000, radius=1.0)
        mesh = MeshBuilder().from_point_cloud(pts, normals=norms)
        sampled = MeshBuilder.sample_points(mesh, n=500)
        radii = np.linalg.norm(sampled, axis=1)
        assert np.abs(radii.mean() - 1.0) < 0.3
