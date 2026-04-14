"""Concrete SensorStream implementations for offline / demo usage."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from engine_grounder.streams.sensor_stream import SensorStream


class MockStream(SensorStream):
    """Loads an RGB-D pair from disk (Redwood-style: .jpg + 16-bit .png)."""

    def __init__(self, data_dir: str = "data/sample_scene"):
        self.rgb_path = os.path.join(data_dir, "color.jpg")
        self.depth_path = os.path.join(data_dir, "depth.png")
        self._connected = False

    def connect(self):
        for p in (self.rgb_path, self.depth_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
        self._connected = True

    def get_frame(self) -> dict:
        if not self._connected:
            self.connect()

        rgb_raw = cv2.imread(self.rgb_path)
        if rgb_raw is None:
            raise FileNotFoundError(f"Could not load RGB: {self.rgb_path}")
        depth_raw = cv2.imread(self.depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Could not load depth: {self.depth_path}")

        rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
        depth_m = depth_raw.astype(np.float32) / 1000.0

        return {"rgb": rgb, "depth": depth_m, "intrinsics": None}

    def close(self):
        self._connected = False

    def get_corrupted_frame(
        self,
        u_min: int = 200, v_min: int = 200,
        u_max: int = 300, v_max: int = 300,
    ) -> dict:
        frame = self.get_frame()
        depth = frame["depth"]
        h, w = depth.shape[:2]
        v_min = max(0, min(v_min, h))
        v_max = max(0, min(v_max, h))
        u_min = max(0, min(u_min, w))
        u_max = max(0, min(u_max, w))
        depth[v_min:v_max, u_min:u_max] = 0.0
        frame["depth"] = depth
        return frame


class BunnyStream(SensorStream):
    """Loads the Stanford Bunny depth + intrinsics from .npy files."""

    def __init__(
        self,
        depth_path: str = "data/bunny_depth.npy",
        intrinsics_path: str = "data/bunny_intrinsics.npy",
    ):
        self.depth_path = depth_path
        self.intrinsics_path = intrinsics_path
        self._connected = False

    def connect(self):
        for p in (self.depth_path, self.intrinsics_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
        self._connected = True

    def get_frame(self) -> dict:
        if not self._connected:
            self.connect()

        depth = np.load(self.depth_path).astype(np.float32)
        fx, fy, cx, cy = np.load(self.intrinsics_path).tolist()

        return {
            "rgb": None,
            "depth": depth,
            "intrinsics": (fx, fy, cx, cy),
        }

    def close(self):
        self._connected = False
