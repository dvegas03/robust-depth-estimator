"""
Generates a simulated pinhole depth map from the Stanford Bunny mesh.

Steps:
  1. Download the mesh via Open3D.
  2. Fill the famous bottom hole via ball-pivoting reconstruction.
  3. Normalise, orient right-side-up, place at CAMERA_DIST.
  4. Z-buffer project to a depth image and save.

Saves to:
    data/bunny_depth.npy       — float32 depth image (H x W), 0 = no reading
    data/bunny_intrinsics.npy  — [fx, fy, cx, cy]  (shape: 4,)

Run once from engine_grounder/:
    python prepare_bunny.py
"""

import numpy as np
import open3d as o3d
import os

# ── Camera / image config ────────────────────────────────────────────────────
IMG_H, IMG_W = 480, 640
FX = FY = 580.0
CX, CY = IMG_W / 2.0, IMG_H / 2.0
CAMERA_DIST = 2.5
N_SAMPLES = 800_000
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_REPO_ROOT, "data")
# ─────────────────────────────────────────────────────────────────────────────


def fill_mesh_holes(mesh):
    """Fill holes in the mesh using Poisson surface reconstruction."""
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=100_000)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(30)

    watertight, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, linear_fit=True
    )
    # Crop to the bounding box of the original mesh to remove Poisson artifacts
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.05, bbox.get_center())
    watertight = watertight.crop(bbox)
    watertight.compute_vertex_normals()
    return watertight


def build_depth_map(points):
    """Z-buffer projection of a 3-D point cloud onto the image plane."""
    z = points[:, 2]
    u = (FX * points[:, 0] / z + CX).astype(np.int32)
    v = (FY * points[:, 1] / z + CY).astype(np.int32)

    in_bounds = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H) & (z > 0)
    u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]

    order = np.argsort(-z)
    depth = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    depth[v[order], u[order]] = z[order].astype(np.float32)
    return depth


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Downloading Stanford Bunny mesh via Open3D …")
    dataset = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)
    mesh.compute_vertex_normals()
    print(f"  Original mesh : {len(mesh.vertices):,} verts, "
          f"{len(mesh.triangles):,} tris")

    print("  Filling bottom hole (Poisson reconstruction) …")
    mesh = fill_mesh_holes(mesh)
    print(f"  Watertight mesh : {len(mesh.vertices):,} verts, "
          f"{len(mesh.triangles):,} tris")

    print(f"  Sampling {N_SAMPLES:,} surface points …")
    pcd = mesh.sample_points_uniformly(number_of_points=N_SAMPLES)
    pts = np.asarray(pcd.points, dtype=np.float64)

    # ── Normalise: centre at origin, fit inside [-1, 1]^3 ────────────────────
    pts -= pts.mean(axis=0)
    pts /= np.abs(pts).max()

    # ── Orient right-side-up ──────────────────────────────────────────────────
    # The raw bunny mesh has ears pointing +Y in object space.
    # In pinhole image coords Y points *down*, so we keep Y as-is (ears up in
    # the image means ears at lower v, which means lower Y in object space
    # after flipping).  We just negate Y to convert from OpenGL to image frame.
    pts[:, 1] *= -1.0

    # ── Place at CAMERA_DIST in front of the camera ──────────────────────────
    pts[:, 2] = CAMERA_DIST + pts[:, 2] * 0.9

    print("  Projecting to depth image …")
    depth = build_depth_map(pts)

    valid = depth > 0
    print(f"  Valid pixel coverage : {valid.mean()*100:.1f} %")
    print(f"  Depth range          : [{depth[valid].min():.3f}, "
          f"{depth[valid].max():.3f}] m")

    depth_path      = os.path.join(OUT_DIR, "bunny_depth.npy")
    intrinsics_path = os.path.join(OUT_DIR, "bunny_intrinsics.npy")
    np.save(depth_path, depth)
    np.save(intrinsics_path, np.array([FX, FY, CX, CY], dtype=np.float32))

    print(f"\nSaved depth map    → {depth_path}")
    print(f"Saved intrinsics   → {intrinsics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
