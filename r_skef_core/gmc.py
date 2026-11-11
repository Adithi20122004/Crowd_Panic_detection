"""
r_skef_core/gmc.py -- RECONSTRUCTED global motion compensation (ego-motion) module.

NOT recovered historical source. Written to satisfy the interface detector.py uses:

    gmc = GlobalMotionCompensator(cfg.gmc_max_features, cfg.gmc_ransac_threshold)
    aligned_prev, diag = gmc.estimate_and_warp(prev_gray, curr_gray)

detector.py then computes optical flow between `aligned_prev` and `curr_gray`, so
camera motion is (approximately) removed from the flow. When enable_gmc is False the
detector never constructs this class and uses the raw previous frame, so the
uncorrected path is untouched.

RECONSTRUCTED DESIGN DECISIONS (none historically verified):
  * Features: Shi-Tomasi corners on the previous frame (max_features corners).
  * Tracking: pyramidal Lucas-Kanade to the current frame.
  * Model: 4-DoF similarity (translation, rotation, uniform scale) fitted with RANSAC
    (cv2.estimateAffinePartial2D); gmc_ransac_threshold is the reprojection threshold in pixels.
  * Fallback: if too few matches or the fit fails, return the unwarped previous frame
    (equivalent to GMC being off for that frame) and report success=False in diag.
  * Known limitation: in a dense moving crowd, most trackable corners may lie on moving
    people, so RANSAC can lock onto crowd motion instead of the background.
"""
from typing import Dict, Tuple

import cv2
import numpy as np

# Reconstructed constants (not config fields; config.py defines none for these).
_QUALITY_LEVEL = 0.01
_MIN_DISTANCE = 8
_BLOCK_SIZE = 7
_LK_WIN = (21, 21)
_LK_MAX_LEVEL = 3
_MIN_MATCHES = 6
# Additional GMC safety guards (experimental validation settings).
_MIN_INLIERS = 8
_MIN_INLIER_RATIO = 0.30

# Plausibility limits for frame-to-frame camera motion.
_MIN_SCALE = 0.90
_MAX_SCALE = 1.10
_MAX_ROTATION_DEG = 5.0
_MAX_TRANSLATION_RATIO = 0.10


class GlobalMotionCompensator:
    def __init__(self, max_features: int, ransac_threshold: float):
        if int(max_features) < 1:
            raise ValueError("max_features must be >= 1")
        if float(ransac_threshold) <= 0:
            raise ValueError("ransac_threshold must be > 0")
        self.max_features = int(max_features)
        self.ransac_threshold = float(ransac_threshold)

    def _fallback(self, prev_gray: np.ndarray, reason: str, n_features: int = 0, n_matches: int = 0
                  ) -> Tuple[np.ndarray, Dict]:
        output_valid = (
            isinstance(prev_gray, np.ndarray)
            and prev_gray.ndim >= 2
        )
        return prev_gray, {
            "success": False, "reason": reason,
            "fallback": True,
            "warp_output_valid": output_valid,
            "affine_estimated": False,
            "affine_estimate_accepted": False,
            "warp_applied": False,
            "estimated_non_identity_transform": False,
            "n_features": n_features, "n_matches": n_matches, "n_inliers": 0,
            "inlier_ratio": 0.0, "tx": 0.0, "ty": 0.0, "rotation_deg": 0.0, "scale": 1.0,
        }

    def estimate_and_warp(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Return a candidate-aligned frame and diagnostics, not an accuracy guarantee.

        ``success`` means the affine estimate passed local guards and was warped.
        It does not establish that the estimate represents camera motion.
        """

        h, w = prev_gray.shape[:2]

        if curr_gray.shape[:2] != (h, w):
            return self._fallback(prev_gray, "frame_size_mismatch")

        p0 = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_features,
            qualityLevel=_QUALITY_LEVEL,
            minDistance=_MIN_DISTANCE,
            blockSize=_BLOCK_SIZE,
        )

        if p0 is None or len(p0) < _MIN_MATCHES:
            return self._fallback(
                prev_gray, "too_few_features",
                0 if p0 is None else len(p0)
            )

        n_features = len(p0)

        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            p0,
            None,
            winSize=_LK_WIN,
            maxLevel=_LK_MAX_LEVEL,
        )

        if p1 is None or status is None:
            return self._fallback(prev_gray, "tracking_failed", n_features)

        ok = status.reshape(-1) == 1
        src = p0.reshape(-1, 2)[ok]
        dst = p1.reshape(-1, 2)[ok]

        if len(src) < _MIN_MATCHES:
            return self._fallback(
                prev_gray, "too_few_matches", n_features, len(src)
            )

        M, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )

        if M is None or inliers is None:
            return self._fallback(
                prev_gray, "ransac_failed", n_features, len(src)
            )

        n_inliers = int(inliers.sum())
        inlier_ratio = n_inliers / len(src)
        tx = float(M[0, 2]) if np.isfinite(M[0, 2]) else 0.0
        ty = float(M[1, 2]) if np.isfinite(M[1, 2]) else 0.0
        rotation_deg = float(
            np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        ) if np.all(np.isfinite(M[:2, :2])) else 0.0
        scale = float(np.hypot(M[0, 0], M[1, 0])) if np.all(np.isfinite(M[:2, :2])) else 1.0
        non_identity_estimate = (
            abs(tx) > 1e-6
            or abs(ty) > 1e-6
            or abs(rotation_deg) > 1e-6
            or abs(scale - 1.0) > 1e-6
        )

        # Guard 1: Reject weak RANSAC fits.
        if n_inliers < _MIN_INLIERS or inlier_ratio < _MIN_INLIER_RATIO:
            aligned_prev, diag = self._fallback(
                prev_gray, "insufficient_ransac_inliers",
                n_features, len(src)
            )
            diag.update({
                "affine_estimated": True,
                "n_inliers": n_inliers,
                "inlier_ratio": float(inlier_ratio),
                "tx": tx,
                "ty": ty,
                "rotation_deg": rotation_deg,
                "scale": scale,
                "estimated_non_identity_transform": non_identity_estimate,
            })
            return aligned_prev, diag

        # Guard 2: Reject invalid affine transformations.
        if not np.all(np.isfinite(M)):
            aligned_prev, diag = self._fallback(
                prev_gray, "non_finite_transform",
                n_features, len(src)
            )
            diag.update({
                "affine_estimated": True,
                "n_inliers": n_inliers,
                "inlier_ratio": float(inlier_ratio),
            })
            return aligned_prev, diag

        max_translation_x = w * _MAX_TRANSLATION_RATIO
        max_translation_y = h * _MAX_TRANSLATION_RATIO

        if (
            not _MIN_SCALE <= scale <= _MAX_SCALE
            or abs(rotation_deg) > _MAX_ROTATION_DEG
            or abs(tx) > max_translation_x
            or abs(ty) > max_translation_y
        ):
            aligned_prev, diag = self._fallback(
                prev_gray, "unrealistic_transform",
                n_features, len(src)
            )
            diag.update({
                "affine_estimated": True,
                "n_inliers": n_inliers,
                "inlier_ratio": float(inlier_ratio),
                "tx": tx,
                "ty": ty,
                "rotation_deg": rotation_deg,
                "scale": scale,
                "estimated_non_identity_transform": non_identity_estimate,
            })
            return aligned_prev, diag

        aligned_prev = cv2.warpAffine(
            prev_gray,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        diag = {
            "success": True,
            "reason": "ok",
            "fallback": False,
            "warp_output_valid": True,
            "affine_estimated": True,
            "affine_estimate_accepted": True,
            "warp_applied": True,
            "estimated_non_identity_transform": non_identity_estimate,
            "n_features": n_features,
            "n_matches": int(len(src)),
            "n_inliers": n_inliers,
            "inlier_ratio": float(inlier_ratio),
            "tx": tx,
            "ty": ty,
            "rotation_deg": rotation_deg,
            "scale": scale,
        }

        return aligned_prev, diag