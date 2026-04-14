"""
Quantitative metric functions for the Robust Depth Estimator pipeline.

All functions are pure NumPy / SciPy and accept plain arrays.
No pipeline imports here — this module is dependency-free.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# Depth estimation accuracy
# ══════════════════════════════════════════════════════════════════════════════

def z_error_cm(z_est: float, z_true: float) -> float:
    """Absolute Z estimation error in centimetres."""
    return abs(z_est - z_true) * 100.0


def depth_rmse(
    depth_pred: np.ndarray,
    depth_true: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Root Mean Squared Error over valid pixels.

    Parameters
    ----------
    depth_pred  : filtered / estimated depth map (H x W float32)
    depth_true  : ground-truth depth map (same shape)
    mask        : boolean array; defaults to pixels > 0 in BOTH maps
    """
    if mask is None:
        mask = (depth_pred > 0) & (depth_true > 0)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((depth_pred[mask] - depth_true[mask]) ** 2)))


def depth_mae(
    depth_pred: np.ndarray,
    depth_true: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Mean Absolute Error over valid pixels."""
    if mask is None:
        mask = (depth_pred > 0) & (depth_true > 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(depth_pred[mask] - depth_true[mask])))


def depth_bias(
    depth_pred: np.ndarray,
    depth_true: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Signed mean bias: positive = overestimation, negative = underestimation.
    """
    if mask is None:
        mask = (depth_pred > 0) & (depth_true > 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(depth_pred[mask] - depth_true[mask]))


def rmse_improvement_ratio(rmse_before: float, rmse_after: float) -> float:
    """
    How much the filter improved RMSE.
    Value > 1 means RMSE decreased (good).  1.5 → 50% improvement.
    """
    if rmse_after == 0 or np.isnan(rmse_after):
        return float("nan")
    return rmse_before / rmse_after


# ══════════════════════════════════════════════════════════════════════════════
# Outlier / anomaly detection quality
# ══════════════════════════════════════════════════════════════════════════════

def outlier_metrics(
    predicted_mask: np.ndarray,
    true_outlier_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Precision / Recall / F1 / FPR for binary outlier detection.

    Evaluation is restricted to *valid* (non-void) pixels because voids are
    handled before outlier detection and should not count as TP/FP/FN.

    Returns
    -------
    dict with:
        precision           : TP / (TP + FP)
        recall              : TP / (TP + FN)
        f1                  : harmonic mean of precision and recall
        false_positive_rate : FP / (FP + TN)  — valid inliers wrongly flagged
        tp, fp, fn, tn      : raw counts
    """
    pred = predicted_mask.astype(bool) & valid_mask.astype(bool)
    true = true_outlier_mask.astype(bool) & valid_mask.astype(bool)

    tp = int(( pred &  true).sum())
    fp = int(( pred & ~true).sum())
    fn = int((~pred &  true).sum())
    tn = int((~pred & ~true & valid_mask.astype(bool)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def inlier_retention_rate(
    predicted_outlier_mask: np.ndarray,
    true_outlier_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    """
    Fraction of true *inlier* valid pixels that were NOT falsely flagged.

    A value of 0.90 means 90% of clean pixels were kept — the filter threw
    away only 10% of the signal.
    """
    true_inlier = valid_mask.astype(bool) & ~true_outlier_mask.astype(bool)
    if not true_inlier.any():
        return float("nan")
    false_flags = predicted_outlier_mask.astype(bool) & true_inlier
    return float(1.0 - false_flags.sum() / true_inlier.sum())


# ══════════════════════════════════════════════════════════════════════════════
# Noise model accuracy
# ══════════════════════════════════════════════════════════════════════════════

def sigma_recovery_rel_error(
    sigma_hat: float,
    sigma_true: float,
) -> float:
    """Relative error of the adaptive σ̂ estimate vs injected noise σ."""
    if sigma_true <= 0:
        return float("nan")
    return abs(sigma_hat - sigma_true) / sigma_true


# ══════════════════════════════════════════════════════════════════════════════
# 3-D geometry quality
# ══════════════════════════════════════════════════════════════════════════════

def chamfer_hausdorff(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    n_sample: int = 5_000,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Symmetric Chamfer distance and Hausdorff distance between two 3-D
    point clouds.

    Both are measured in the same units as the input coordinates (metres).

    Parameters
    ----------
    pts_a, pts_b : (N, 3) and (M, 3) float arrays
    n_sample     : subsample each cloud to at most this many points

    Returns
    -------
    (chamfer_l2, hausdorff)
        chamfer_l2  : mean nearest-neighbour L2 averaged over both directions
        hausdorff   : max nearest-neighbour L2 over both directions
    """
    from scipy.spatial import KDTree

    rng = np.random.default_rng(seed)
    if len(pts_a) > n_sample:
        pts_a = pts_a[rng.choice(len(pts_a), n_sample, replace=False)]
    if len(pts_b) > n_sample:
        pts_b = pts_b[rng.choice(len(pts_b), n_sample, replace=False)]

    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("nan"), float("nan")

    d_a2b, _ = KDTree(pts_b).query(pts_a, k=1, workers=-1)
    d_b2a, _ = KDTree(pts_a).query(pts_b, k=1, workers=-1)

    chamfer   = float(0.5 * (d_a2b.mean() + d_b2a.mean()))
    hausdorff = float(max(d_a2b.max(), d_b2a.max()))
    return chamfer, hausdorff


def point_cloud_coverage(
    pts_reconstructed: np.ndarray,
    pts_reference: np.ndarray,
    radius: float = 0.05,
) -> float:
    """
    Surface coverage: fraction of reference points that have at least one
    reconstructed point within *radius* metres.

    A value of 0.90 means 90% of the reference surface is covered.
    """
    from scipy.spatial import KDTree

    if len(pts_reconstructed) == 0 or len(pts_reference) == 0:
        return 0.0

    d, _ = KDTree(pts_reconstructed).query(pts_reference, k=1, workers=-1)
    return float((d <= radius).mean())
