# src/annotator.py
"""
2-panel annotated output image.

Panel 1 (left)  — Original with SAM3 smoke overlay
Panel 2 (right) — Inpainted result (pixel mask) with opacity result
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def save_annotated(
    original_bgr: np.ndarray,
    inpainted_pixel,          # np.ndarray or None
    smoke_mask_dict: dict,
    result: dict,
    pixel_mask: np.ndarray,
    output_path: str,
) -> None:
    H, W = original_bgr.shape[:2]
    canvas = np.zeros((H, W * 2, 3), dtype=np.uint8)

    # ── Panel 1: original + smoke overlay ─────────────────────────────────────
    p1  = original_bgr.copy()
    seg = smoke_mask_dict["segmentation"]
    col = cfg.REGION_COLORS["smoke_light"]
    ov  = p1.copy()
    ov[seg] = col
    p1  = cv2.addWeighted(ov, 0.4, p1, 0.6, 0)
    ctrs, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p1, ctrs, -1, col, cfg.ANNOTATION_THICKNESS)
    _banner(p1, "Original + Smoke Region", None, None)
    canvas[:H, :W] = p1

    # ── Panel 2: pixel mask inpainted result ──────────────────────────────────
    if inpainted_pixel is not None:
        p2 = inpainted_pixel.copy()
        mc, _ = cv2.findContours(pixel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p2, mc, -1, (0, 255, 255), cfg.ANNOTATION_THICKNESS)
    else:
        p2 = np.zeros_like(original_bgr)
        cv2.putText(p2, "Inpainting failed", (20, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)
    _banner(p2, "Pixel Mask (SAM3)", result["opacity_pct"], result["ringelman"])
    canvas[:H, W:] = p2

    # Vertical divider
    canvas[:, W - 1 : W + 1] = (60, 60, 60)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"[Annotator] Saved → {output_path}")


def _banner(img: np.ndarray, title: str, opacity_pct, ringelman) -> None:
    H, W = img.shape[:2]
    cv2.rectangle(img, (0, H - 70), (W, H), (20, 20, 20), -1)
    cv2.putText(img, title, (10, H - 46),
                cv2.FONT_HERSHEY_SIMPLEX, cfg.ANNOTATION_FONT_SCALE * 0.65,
                (200, 200, 200), cfg.ANNOTATION_THICKNESS)
    if opacity_pct is not None:
        label = f"Opacity: {opacity_pct:.1f}%   |   Ringelman: Grade {ringelman}"
        cv2.putText(img, label, (10, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.ANNOTATION_FONT_SCALE * 0.65,
                    (0, 255, 100), cfg.ANNOTATION_THICKNESS)
    else:
        cv2.putText(img, "Result: N/A", (10, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.ANNOTATION_FONT_SCALE * 0.65,
                    (0, 100, 220), cfg.ANNOTATION_THICKNESS)
