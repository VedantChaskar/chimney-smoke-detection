"""
Module 5: Visualization & Output.

Generates annotated images and multi-panel debug figures.
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# Ringelmann badge colours (green→yellow→orange→red)
_RING_COLORS = [
    (0,   200, 0),    # R0 green
    (100, 220, 0),    # R1 yellow-green
    (220, 220, 0),    # R2 yellow
    (255, 140, 0),    # R3 orange
    (220,  60, 0),    # R4 dark orange
    (180,   0, 0),    # R5 red
]


def _apply_alpha_heatmap(
    image_bgr: np.ndarray,
    alpha_map: np.ndarray,
    smoke_mask: np.ndarray,
    blend: float = 0.55,
) -> np.ndarray:
    """Overlay a jet-colourmap heatmap of alpha over the original image."""
    h, w = image_bgr.shape[:2]
    norm_alpha = np.clip(alpha_map, 0.0, 1.0)

    # jet colormap via matplotlib
    cmap = plt.get_cmap("jet")
    rgba = (cmap(norm_alpha) * 255).astype(np.uint8)          # (H, W, 4)
    heat_bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)

    overlay = image_bgr.copy().astype(np.float32)
    mask = smoke_mask.astype(bool)

    overlay[mask] = (
        (1.0 - blend) * image_bgr[mask].astype(np.float32)
        + blend * heat_bgr[mask].astype(np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _draw_smoke_boundary(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw the smoke mask boundary on a copy of the image."""
    out = image_bgr.copy()
    m   = binary_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def _ringelmann_badge(ringelmann: int) -> tuple:
    """Return BGR colour for a Ringelmann badge."""
    r, g, b = _RING_COLORS[min(ringelmann, 5)]
    return (b, g, r)  # BGR


def annotate_overlay(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    alpha_map: np.ndarray,
    result: dict,
) -> np.ndarray:
    """
    Single-image overlay:
      - Alpha heatmap blended over smoke region
      - Green smoke boundary
      - Ringelmann badge in top-left corner

    Args:
        image_bgr:   (H, W, 3) BGR.
        binary_mask: (H, W) bool.
        alpha_map:   (H, W) float32 opacity map.
        result:      Fusion result dict (keys: ringelmann, fused_alpha, confidence).

    Returns:
        (H, W, 3) BGR annotated image.
    """
    out = _apply_alpha_heatmap(image_bgr, alpha_map, binary_mask)
    out = _draw_smoke_boundary(out, binary_mask)

    ring = result["ringelmann"]
    alpha_str = f"R{ring}  α={result['fused_alpha']:.2f}  [{result['confidence']}]"
    badge_color = _ringelmann_badge(ring)

    # Badge background
    text_size, _ = cv2.getTextSize(alpha_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pad = 8
    cv2.rectangle(
        out,
        (10 - pad, 10 - pad),
        (10 + text_size[0] + pad, 10 + text_size[1] + pad),
        badge_color,
        -1,
    )
    cv2.putText(
        out, alpha_str,
        (10, 10 + text_size[1]),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


def create_output_panel(
    original: np.ndarray,
    binary_mask: np.ndarray,
    alpha_map: np.ndarray,
    result: dict,
    save_path: str | Path,
    dpi: int = 120,
) -> None:
    """
    Save a 2×2 panel figure:
        Top-left:     Original image
        Top-right:    Smoke mask overlay (green boundary)
        Bottom-left:  Alpha heatmap overlaid on image
        Bottom-right: Ringelmann gauge + text summary

    Args:
        original:    (H, W, 3) BGR image.
        binary_mask: (H, W) bool mask.
        alpha_map:   (H, W) float32 opacity.
        result:      Fusion result dict.
        save_path:   Output file path (.png or .jpg).
        dpi:         Output DPI.
    """
    rgb        = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    mask_vis   = _draw_smoke_boundary(original, binary_mask)
    mask_vis_r = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
    heat_vis   = _apply_alpha_heatmap(original, alpha_map, binary_mask)
    heat_vis_r = cv2.cvtColor(heat_vis, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi)

    # Top-left: original
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original Image", fontsize=12)
    axes[0, 0].axis("off")

    # Top-right: mask boundary
    axes[0, 1].imshow(mask_vis_r)
    axes[0, 1].set_title("Smoke Mask (green boundary)", fontsize=12)
    axes[0, 1].axis("off")

    # Bottom-left: heatmap
    im = axes[1, 0].imshow(heat_vis_r)
    axes[1, 0].set_title("Opacity Heatmap (jet)", fontsize=12)
    axes[1, 0].axis("off")
    # Colour-bar
    sm = ScalarMappable(cmap="jet", norm=Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Opacity α")

    # Bottom-right: Ringelmann gauge
    ax = axes[1, 1]
    ax.axis("off")

    ring      = result["ringelmann"]
    fused     = result["fused_alpha"]
    conf      = result["confidence"]
    per       = result.get("per_method", {})
    seg_m     = result.get("segmentation_method", "unknown")

    badge_rgb = tuple(c / 255 for c in _RING_COLORS[ring])

    # Large Ringelmann number
    ax.text(0.5, 0.78, f"R{ring}", ha="center", va="center",
            fontsize=72, fontweight="bold", color=badge_rgb,
            transform=ax.transAxes)
    ax.text(0.5, 0.62, f"Ringelmann {ring}", ha="center", va="center",
            fontsize=14, transform=ax.transAxes)

    lines = [
        f"Fused α : {fused:.3f}",
        f"Confidence: {conf}",
        f"Segmentation: {seg_m}",
        "",
        "Per-method α:",
        f"  Dark channel : {per.get('dark_channel',  0.0):.3f}",
        f"  Contrast     : {per.get('contrast',      0.0):.3f}",
        f"  Color collapse: {per.get('color_collapse', 0.0):.3f}",
    ]
    ax.text(0.05, 0.50, "\n".join(lines), ha="left", va="top",
            fontsize=10, family="monospace", transform=ax.transAxes)

    # Ringelmann scale legend
    scale_labels = [f"R{i}" for i in range(6)]
    handles = [
        mpatches.Patch(
            color=[c / 255 for c in _RING_COLORS[i]],
            label=scale_labels[i],
        )
        for i in range(6)
    ]
    ax.legend(handles=handles, loc="lower center",
              ncol=6, fontsize=9, title="Ringelmann scale",
              bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)
