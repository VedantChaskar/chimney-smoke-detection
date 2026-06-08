# src/annotator.py
"""
Draws annotated output image showing SAM3 regions + opacity results.
"""

from __future__ import annotations

import os
import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def save_annotated_image(
    smoke_bgr: np.ndarray,
    best_regions: dict,
    result: dict,
    output_path: str,
) -> None:
    """
    Create and save an annotated version of the smoke image.

    Annotations:
        - Semi-transparent coloured overlay per detected region
        - Contour outline around each region
        - Region-type label at centroid
        - Info panel (top-left): opacity %, Ringelman grade, formula, smoke type
    """
    annotated = smoke_bgr.copy()

    # ── Region overlays ───────────────────────────────────────────────────────
    for region_type, mask_dict in best_regions.items():
        if mask_dict is None:
            continue

        color = cfg.REGION_COLORS.get(region_type, (128, 128, 128))
        seg   = mask_dict["segmentation"]

        # Semi-transparent fill
        overlay = annotated.copy()
        overlay[seg] = color
        annotated    = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

        # Contour outline
        contours, _ = cv2.findContours(
            seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, color, cfg.ANNOTATION_THICKNESS)

        # Label at centroid
        ys, xs = np.where(seg)
        if len(xs) > 0:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            cv2.putText(
                annotated, region_type, (max(cx - 50, 0), cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg.ANNOTATION_FONT_SCALE * 0.55,
                color,
                cfg.ANNOTATION_THICKNESS,
            )

    # ── Info panel ────────────────────────────────────────────────────────────
    opacity_str   = (f"{result['opacity_pct']:.1f}%"
                     if result["opacity_pct"] is not None else "N/A")
    ringelman_str = (str(result["ringelman"])
                     if result["ringelman"] is not None else "N/A")

    lines = [
        f"Opacity:    {opacity_str}",
        f"Ringelman:  Grade {ringelman_str}",
        f"Smoke Type: {result['smoke_type']}",
        f"Formula:    {result['formula_used']}",
    ]
    if result.get("reason", "OK") != "OK":
        lines.append(f"Note: {result['reason']}")

    panel_h = 38 * (len(lines) + 1)
    panel_w = 500
    cv2.rectangle(annotated, (10, 10), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.rectangle(annotated, (10, 10), (panel_w, panel_h), (180, 180, 180), 2)

    for i, line in enumerate(lines):
        color = (0, 255, 0) if i < 2 else (200, 200, 200)
        cv2.putText(
            annotated, line, (20, 48 + i * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            cfg.ANNOTATION_FONT_SCALE * 0.6,
            color,
            cfg.ANNOTATION_THICKNESS,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    print(f"[Annotator] Saved → {output_path}")
