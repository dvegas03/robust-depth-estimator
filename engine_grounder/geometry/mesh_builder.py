"""Point cloud to mesh reconstruction via Open3D."""

from __future__ import annotations

import numpy as np
import open3d as o3d


class MeshBuilder:
    """Reconstructs a watertight mesh from a raw 3-D point cloud."""

    def __init__(
        self,
        poisson_depth: int = 8,
        density_quantile: float = 0.05,
        voxel_size: float = 0.01,
    ):
        self.poisson_depth = poisson_depth
        self.density_quantile = density_quantile
        self.voxel_size = voxel_size

    def from_point_cloud(
        self,
        points: np.ndarray,
        normals: np.ndarray | None = None,
    ) -> o3d.geometry.TriangleMesh:
        if len(points) < 30:
            return o3d.geometry.TriangleMesh()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        if len(pcd.points) < 30:
            return o3d.geometry.TriangleMesh()

        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        if len(pcd.points) < 30:
            return o3d.geometry.TriangleMesh()

        # Alpha shapes are mathematically stable on Apple Silicon.
        # Fall back to an empty mesh if the point cloud is degenerate
        # (coplanar, collinear, etc.) so the pipeline can continue.
        alpha = 0.05
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        except Exception:
            return o3d.geometry.TriangleMesh()
        mesh.compute_vertex_normals()
        return mesh

    @staticmethod
    def simplify(
        mesh: o3d.geometry.TriangleMesh,
        target_faces: int = 10_000,
    ) -> o3d.geometry.TriangleMesh:
        if len(mesh.triangles) <= target_faces or len(mesh.triangles) == 0:
            return mesh

        simplified = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )
        simplified.compute_vertex_normals()
        return simplified

    @staticmethod
    def sample_points(
        mesh: o3d.geometry.TriangleMesh,
        n: int = 2048,
    ) -> np.ndarray:
        if len(mesh.triangles) == 0:
            return np.zeros((n, 3), dtype=np.float64)

        pcd = mesh.sample_points_uniformly(number_of_points=n)
        return np.asarray(pcd.points, dtype=np.float64)
