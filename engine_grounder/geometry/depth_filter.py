"""
Industry-grade depth map filtering pipeline.

Outlier detection: **adaptive bilateral-residual analysis**.

Algorithm
---------
1. Fill voids with local-mean of valid neighbours (context for the bilateral).
2. Median pre-filter (3×3) defuses isolated extreme spikes so the bilateral
   filter is not biased by pepper noise (self-weight dominance issue).
3. Edge-preserving bilateral smooth → per-pixel residuals.
4. **Adaptive sigma map**: depth-binned Q50 estimator recovers the true
   sensor noise σ(Z) directly from the residuals — no manual ``noise_sigma``
   required.  Uses only pixels with sufficient valid-neighbour density so
   void-boundary artifacts don't contaminate the estimate.
5. Flag pixels where |residual| > ``outlier_k × σ_hat(Z)``.
6. Optional global IQR safety-net (legacy, kept as utility).

Processing stages (full pipeline):
  1. Void Masking                — reject 0, negative, NaN, ±inf
  2. Adaptive Outlier Rejection — adaptive bilateral-residual
  3. Morphological Closing       — fill small void clusters
  4. Depth Inpainting            — OpenCV Telea / Navier-Stokes
  5. Bilateral Smoothing         — final edge-preserving denoise
"""

import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter
from typing import Optional, Tuple


class RobustDepthEstimator:
    """
    Parameters
    ----------
    tau_threshold       : Minimum valid-pixel ratio before perimeter fallback.
    outlier_k           : Sigma multiplier for outlier gating (3.0 = 99.7%
                          of a Gaussian; tighter = more aggressive).
    noise_sigma_floor   : Hard minimum on the estimated σ to prevent the
                          threshold collapsing to zero in noise-free regions.
    extreme_threshold   : Absolute deviation (m) above which a pixel is an
                          obvious spike (defused via median pre-filter).
    sigma_bins          : Number of depth bins for the adaptive σ(Z) model.
    density_min         : Minimum local valid-pixel fraction [0,1] required
                          for a pixel to contribute to the σ estimate.
    density_kernel      : Kernel size (px) for local density computation.
    bilateral_d         : Bilateral filter diameter.
    bilateral_sigma_color : σ in depth space (m).
    bilateral_sigma_space : σ in pixel space.
    morph_kernel        : Morphological closing kernel size (px).
    inpaint_radius      : OpenCV inpainting radius (px).
    """

    def __init__(
        self,
        tau_threshold: float = 0.10,
        outlier_k: float = 3.0,
        noise_sigma_floor: float = 0.002,
        extreme_threshold: float = 0.5,
        sigma_bins: int = 20,
        density_min: float = 0.30,
        density_kernel: int = 11,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 0.04,
        bilateral_sigma_space: float = 4.5,
        morph_kernel: int = 5,
        inpaint_radius: int = 5,
    ):
        self.tau = tau_threshold
        self.outlier_k = outlier_k
        self.noise_sigma_floor = noise_sigma_floor
        self.extreme_threshold = extreme_threshold
        self.sigma_bins = sigma_bins
        self.density_min = density_min
        self.density_kernel = density_kernel
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.morph_kernel = morph_kernel
        self.inpaint_radius = inpaint_radius

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1 — Void Masking
    # ══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def void_mask(depth: np.ndarray) -> np.ndarray:
        """Returns boolean mask: True where the pixel is *valid*."""
        return np.isfinite(depth) & (depth > 0.0)

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _void_filled(self, depth: np.ndarray) -> np.ndarray:
        """Fill voids with local mean of valid neighbours."""
        valid = self.void_mask(depth)
        d64 = np.where(valid, depth, 0.0).astype(np.float64)
        vf = valid.astype(np.float64)
        k = self.bilateral_d
        local_sum   = uniform_filter(d64, size=k, mode="constant")
        local_count = uniform_filter(vf,  size=k, mode="constant")
        with np.errstate(divide="ignore", invalid="ignore"):
            local_mean = np.where(
                local_count > 0, local_sum / local_count, 0.0
            ).astype(np.float32)
        return np.where(valid, depth.astype(np.float32), local_mean)

    def _local_density(self, valid: np.ndarray) -> np.ndarray:
        """Return per-pixel fraction of valid neighbours in density_kernel."""
        return uniform_filter(
            valid.astype(np.float64), size=self.density_kernel, mode="constant"
        ).astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # Adaptive σ(Z) estimation — core of the adaptive algorithm
    # ══════════════════════════════════════════════════════════════════════════
    def estimate_sigma_map(
        self,
        depth: np.ndarray,
        residuals: np.ndarray,
        density: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a depth-dependent noise model σ(Z) from per-pixel bilateral
        residuals and return a per-pixel adaptive threshold map.

        Algorithm
        ---------
        1. Divide the valid depth range into ``sigma_bins`` depth bins.
        2. Within each bin, compute Q50(|residuals|) from pixels that have
           sufficient valid-neighbour density (≥ ``density_min``).
        3. For half-normal |N(0, σ)|: median = σ × 0.6745, so:
               σ_hat_bin = Q50 / 0.6745
        4. Interpolate across all bins; fill empty bins from neighbours.
        5. Clamp to [noise_sigma_floor, ∞).
        6. Per-pixel threshold = ``outlier_k × σ_hat(depth)``.

        Returns
        -------
        sigma_map  : HxW float32 — estimated σ at every pixel (metres).
        thresh_map : HxW float32 — adaptive threshold = outlier_k × sigma_map.
        """
        valid = self.void_mask(depth)
        if density is None:
            density = self._local_density(valid)

        # Only use contextually reliable pixels for the estimate
        good = valid & (density >= self.density_min)
        depths_g  = depth[good].astype(np.float64)
        resids_g  = residuals[good].astype(np.float64)

        d_lo = float(depths_g.min()) if depths_g.size > 0 else 0.0
        d_hi = float(depths_g.max()) if depths_g.size > 0 else 1.0
        if d_hi <= d_lo:
            d_hi = d_lo + 1.0

        bins = np.linspace(d_lo, d_hi, self.sigma_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_sigma = np.full(self.sigma_bins, np.nan)

        for i in range(self.sigma_bins):
            mask = (depths_g >= bins[i]) & (depths_g < bins[i + 1])
            if mask.sum() < 5:
                continue
            # Q50 of |residual| for half-normal: median = σ × 0.6745
            q50 = np.median(resids_g[mask])
            bin_sigma[i] = q50 / 0.6745

        # Interpolate across empty bins
        valid_bins = ~np.isnan(bin_sigma)
        if valid_bins.any():
            idx = np.arange(self.sigma_bins)
            bin_sigma = np.interp(idx, idx[valid_bins], bin_sigma[valid_bins])
        else:
            bin_sigma[:] = self.noise_sigma_floor

        # Clamp to floor
        bin_sigma = np.maximum(bin_sigma, self.noise_sigma_floor)

        # Map each pixel to its bin's σ
        pixel_depth = depth.astype(np.float64).ravel()
        bin_idx = np.clip(
            np.searchsorted(bins[1:], pixel_depth, side="left"),
            0, self.sigma_bins - 1,
        )
        sigma_map = bin_sigma[bin_idx].astype(np.float32).reshape(depth.shape)
        thresh_map = (self.outlier_k * sigma_map).astype(np.float32)
        return sigma_map, thresh_map

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2 — Adaptive Bilateral-Residual Outlier Detection
    # ══════════════════════════════════════════════════════════════════════════
    def bilateral_outlier_mask(
        self, depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Adaptive outlier detection — returns (outlier_mask, sigma_map, thresh_map).

        Two-pass approach:
        1. Median pre-filter (3×3) defuses extreme isolated spikes.
        2. Bilateral residual with depth-adaptive threshold.

        The ``thresh_map`` is exposed so callers / the visualiser can display
        how aggressively the filter is operating at each pixel.
        """
        valid = self.void_mask(depth)
        if not valid.any():
            zeros = np.zeros_like(valid)
            return zeros, zeros.astype(np.float32), zeros.astype(np.float32)

        density  = self._local_density(valid)
        filled   = self._void_filled(depth)

        # Pass 1: median pre-filter — defuse isolated extreme spikes
        local_med = median_filter(filled, size=3).astype(np.float32)
        extreme   = valid & (np.abs(filled - local_med) > self.extreme_threshold)
        defused   = filled.copy()
        defused[extreme] = local_med[extreme]

        # Pass 2: edge-preserving bilateral reference surface
        smoothed = cv2.bilateralFilter(
            defused, self.bilateral_d,
            self.bilateral_sigma_color, self.bilateral_sigma_space,
        )
        residuals = np.abs(depth.astype(np.float32) - smoothed)

        # Adaptive threshold from depth-binned σ(Z) model
        sigma_map, thresh_map = self.estimate_sigma_map(depth, residuals, density)

        bilateral_out = valid & ~extreme & (residuals > thresh_map)
        outlier_mask  = extreme | bilateral_out
        return outlier_mask, sigma_map, thresh_map

    def bilateral_filter(self, depth: np.ndarray) -> np.ndarray:
        """Zero out adaptively-detected outliers.  Returns a copy."""
        mask, _, _ = self.bilateral_outlier_mask(depth)
        out = depth.copy()
        out[mask] = 0.0
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # Legacy — Global IQR (kept as utility; not used in default pipeline)
    # ══════════════════════════════════════════════════════════════════════════
    def iqr_filter(self, depth: np.ndarray, iqr_mult: float = 3.0) -> np.ndarray:
        """Zero out global IQR outliers (safety-net).  Returns a copy."""
        out = depth.copy()
        valid = self.void_mask(out)
        vals = out[valid]
        if vals.size == 0:
            return out
        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1
        lo  = q1 - iqr_mult * iqr
        hi  = q3 + iqr_mult * iqr
        out[valid & ((out < lo) | (out > hi))] = 0.0
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3 — Morphological Closing
    # ══════════════════════════════════════════════════════════════════════════
    def morph_close(self, depth: np.ndarray) -> np.ndarray:
        if self.morph_kernel <= 0:
            return depth.copy()

        valid  = self.void_mask(depth).astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
        )
        closed_mask = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel)
        newly_valid = (closed_mask == 1) & (valid == 0)
        if not newly_valid.any():
            return depth.copy()

        out = depth.copy()
        pad_d = np.pad(out, self.morph_kernel, mode="constant", constant_values=0)
        pad_v = np.pad(
            self.void_mask(out).astype(np.float32), self.morph_kernel,
            mode="constant", constant_values=0,
        )
        k = self.morph_kernel
        for y, x in zip(*np.nonzero(newly_valid)):
            patch = pad_d[y : y + 2*k+1, x : x + 2*k+1]
            vmask = pad_v[y : y + 2*k+1, x : x + 2*k+1]
            n = vmask.sum()
            if n > 0:
                out[y, x] = (patch * vmask).sum() / n
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4 — Depth Inpainting
    # ══════════════════════════════════════════════════════════════════════════
    def inpaint(self, depth: np.ndarray, method: str = "telea") -> np.ndarray:
        void = (~self.void_mask(depth)).astype(np.uint8)
        if void.sum() == 0:
            return depth.copy()

        pos     = depth > 0
        d_min   = depth[pos].min() if pos.any() else 0.0
        d_max   = depth[pos].max() if pos.any() else 1.0
        d_range = d_max - d_min if d_max > d_min else 1.0

        normalised  = ((depth - d_min) / d_range * 255).clip(0, 255).astype(np.uint8)
        flag        = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        inpainted_u8 = cv2.inpaint(normalised, void, self.inpaint_radius, flag)

        restored = inpainted_u8.astype(np.float32) / 255.0 * d_range + d_min
        restored[self.void_mask(depth)] = depth[self.void_mask(depth)]
        return restored

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 5 — Bilateral Smoothing
    # ══════════════════════════════════════════════════════════════════════════
    def bilateral_smooth(self, depth: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(
            depth.astype(np.float32),
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Full Pipeline
    # ══════════════════════════════════════════════════════════════════════════
    def restore(self, depth: np.ndarray) -> np.ndarray:
        """Run the full 5-stage restoration pipeline."""
        d = depth.astype(np.float32)
        d = self.bilateral_filter(d)   # adaptive outlier rejection
        d = self.morph_close(d)
        d = self.inpaint(d)
        d = self.bilateral_smooth(d)
        return d

    # ══════════════════════════════════════════════════════════════════════════
    # Z_est Extraction
    # ══════════════════════════════════════════════════════════════════════════
    def get_stable_z(
        self,
        depth_crop: np.ndarray,
        full_depth_map: Optional[np.ndarray] = None,
        bbox: Optional[Tuple] = None,
    ) -> Optional[float]:
        """
        Robust scalar depth estimate from a 2-D depth crop.

        Uses adaptive bilateral outlier rejection, then returns the median
        of the clean inlier pixels.
        """
        if depth_crop.size == 0:
            return None

        valid_mask = self.void_mask(depth_crop)
        valid_pixels = depth_crop[valid_mask]
        if valid_pixels.size == 0:
            return self._perimeter_fallback(full_depth_map, bbox)

        ratio_valid = valid_pixels.size / depth_crop.size
        if ratio_valid < self.tau:
            return self._perimeter_fallback(full_depth_map, bbox)

        if depth_crop.ndim == 2 and min(depth_crop.shape) >= self.bilateral_d:
            outlier, _, _ = self.bilateral_outlier_mask(depth_crop)
            clean_pixels = depth_crop[valid_mask & ~outlier]
        else:
            clean_pixels = valid_pixels

        if clean_pixels.size == 0:
            clean_pixels = valid_pixels

        return float(np.median(clean_pixels))

    # ── Perimeter fallback ────────────────────────────────────────────────────
    def _perimeter_fallback(self, full_depth_map=None, bbox=None):
        if full_depth_map is None or bbox is None:
            return None

        y_min, y_max, x_min, x_max = bbox
        h, w = full_depth_map.shape[:2]

        pad    = max(y_max - y_min, x_max - x_min) // 2
        py_min = max(y_min - pad, 0)
        py_max = min(y_max + pad, h)
        px_min = max(x_min - pad, 0)
        px_max = min(x_max + pad, w)

        ring = full_depth_map[py_min:py_max, px_min:px_max].copy()
        iy0  = y_min - py_min
        ix0  = x_min - px_min
        ring[iy0 : iy0 + (y_max - y_min), ix0 : ix0 + (x_max - x_min)] = 0.0

        valid = ring[(ring > 0.0) & np.isfinite(ring)]
        if valid.size == 0:
            return None
        return float(np.median(valid))
