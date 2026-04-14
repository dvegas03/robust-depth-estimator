"""Pinhole camera math for 2D-to-3D back-projection.

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = D(u, v)
"""

import numpy as np


class Projector:
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        if fx == 0.0 or fy == 0.0:
            raise ValueError(f"Focal lengths must be non-zero (fx={fx}, fy={fy})")
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def backproject(self, u: int, v: int, depth: float) -> np.ndarray:
        """Back-project a single (u, v, depth) into camera-frame 3D."""
        z = depth
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=np.float64)

    def backproject_depth_map(self, depth: np.ndarray) -> np.ndarray:
        """
        Vectorized back-projection of an entire HxW depth map.

        Returns:
            np.ndarray of shape (H, W, 3) containing [X, Y, Z] per pixel.
        """
        h, w = depth.shape[:2]
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

        z = depth.astype(np.float64)
        x = (u_coords - self.cx) * z / self.fx
        y = (v_coords - self.cy) * z / self.fy

        return np.stack([x, y, z], axis=-1)
