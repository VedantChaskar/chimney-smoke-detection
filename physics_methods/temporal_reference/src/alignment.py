# src/alignment.py
"""
Image registration: aligns a smoke image to the ground truth coordinate space.
Uses ORB keypoints + BFMatcher + RANSAC homography.

If insufficient features are found (camera was stationary enough), returns the
original smoke image unchanged — safe fallback, never raises.
"""

from __future__ import annotations

import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def align_to_ground_truth(
    smoke_bgr: np.ndarray,
    gt_bgr: np.ndarray,
) -> tuple[np.ndarray, bool, int]:
    """
    Warp smoke_bgr into gt_bgr's coordinate space.

    Returns
    -------
    aligned_image : (H, W, 3) — warped smoke image (or original on failure)
    success       : True if homography was applied
    num_matches   : number of good feature matches found
    """
    gt_gray    = cv2.cvtColor(gt_bgr,    cv2.COLOR_BGR2GRAY)
    smoke_gray = cv2.cvtColor(smoke_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=cfg.ALIGNMENT_MAX_FEATURES)
    kp_gt,    desc_gt    = orb.detectAndCompute(gt_gray,    None)
    kp_smoke, desc_smoke = orb.detectAndCompute(smoke_gray, None)

    if (desc_gt is None or desc_smoke is None
            or len(kp_gt) < 4 or len(kp_smoke) < 4):
        print("[Alignment] WARNING: not enough keypoints — using unaligned image")
        return smoke_bgr, False, 0

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc_smoke, desc_gt, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < cfg.ALIGNMENT_MATCH_RATIO * n.distance:
                good.append(m)

    num_matches = len(good)
    if num_matches < cfg.ALIGNMENT_MIN_MATCHES:
        print(f"[Alignment] WARNING: only {num_matches} good matches "
              f"(need {cfg.ALIGNMENT_MIN_MATCHES}) — using unaligned image")
        return smoke_bgr, False, num_matches

    src_pts = np.float32([kp_smoke[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_gt[m.trainIdx].pt    for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        print("[Alignment] WARNING: homography failed — using unaligned image")
        return smoke_bgr, False, num_matches

    h, w    = gt_bgr.shape[:2]
    aligned = cv2.warpPerspective(
        smoke_bgr, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    inliers = int(mask.sum()) if mask is not None else num_matches
    print(f"[Alignment] OK — {num_matches} matches, {inliers} inliers")
    return aligned, True, num_matches
