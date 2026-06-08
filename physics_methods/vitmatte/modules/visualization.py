"""
Module 6: Visualization & Output.

Generates a 2×3 panel figure and a single overlay image.
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ..modules.trimap import visualise_trimap

_RING_COLORS_RGB = [
    (0,   200, 0),
    (100, 220, 0),
    (220, 220, 0),
    (255, 140, 0),
    (220,  60, 0),
    (180,   0, 0),
]


def _ring_color_bgr(r: int) -> tuple:
    g, gg, b = _RING_COLORS_RGB[min(r, 5)]
    return (b, gg, g)


def _ring_color_mpl(r: int) -> tuple:
    rgb = _RING_COLORS_RGB[min(r, 5)]
    return tuple(c / 255 for c in rgb)


def _alpha_heatmap_overlay(
    image_bgr: np.ndarray,
    alpha_map: np.ndarray,
    smoke_mask: np.ndarray,
    blend: float = 0.55,
) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heat_rgba = (cmap(np.clip(alpha_map, 0, 1)) * 255).astype(np.uint8)
    heat_bgr  = cv2.cvtColor(heat_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    out = image_bgr.astype(np.float32)
    mask = smoke_mask.astype(bool)
    out[mask] = ((1 - blend) * out[mask] + blend * heat_bgr[mask].astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_boundary(image_bgr, binary_mask, color=(0, 255, 0), thickness=2):
    out = image_bgr.copy()
    m = binary_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def _ringelmann_gauge_axes(ax, ring: int, result: dict, validation: dict) -> None:
    """Draw the Ringelmann summary panel onto a matplotlib Axes."""
    ax.axis("off")
    badge_color = _ring_color_mpl(ring)

    # Large number
    ax.text(0.5, 0.82, f"R{ring}", ha="center", va="center",
            fontsize=68, fontweight="bold", color=badge_color,
            transform=ax.transAxes)
    ax.text(0.5, 0.69, f"Ringelmann {ring}", ha="center", va="center",
            fontsize=13, transform=ax.transAxes)

    stats = result.get("spatial_stats", {})
    seg_method = result.get("segmentation_method", "unknown")
    quality    = validation.get("overall_quality", "?")
    conf_score = validation.get("confidence_score", 0.0)

    lines = [
        f"Continuous R : {result.get('ringelmann_float', 0.0):.2f}",
        f"Rep. alpha   : {result.get('representative_alpha', 0.0):.3f}",
        f"Method       : {result.get('method_used', '?')}",
        f"Quality      : {quality}  (score={conf_score:.2f})",
        f"Segmentation : {seg_method}",
        "",
        "Spatial stats (smoke pixels):",
        f"  mean α  : {stats.get('mean_alpha',   0):.3f}",
        f"  p75  α  : {stats.get('p75_alpha',    0):.3f}",
        f"  p90  α  : {stats.get('p90_alpha',    0):.3f}",
        f"  std  α  : {stats.get('std_alpha',    0):.3f}",
        f"  coverage: {stats.get('coverage_fraction', 0)*100:.1f}%",
    ]
    warnings = validation.get("warnings", [])
    if warnings:
        lines += ["", "⚠ Warnings:"] + [f"  • {w[:55]}" for w in warnings[:3]]

    ax.text(0.03, 0.62, "\n".join(lines), ha="left", va="top",
            fontsize=8.5, family="monospace", transform=ax.transAxes)

    # Ringelmann scale legend
    handles = [
        mpatches.Patch(color=_ring_color_mpl(i), label=f"R{i}")
        for i in range(6)
    ]
    ax.legend(handles=handles, loc="lower center", ncol=6,
              fontsize=8, title="Ringelmann scale",
              bbox_to_anchor=(0.5, 0.0))


def annotate_overlay(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    alpha_map: np.ndarray,
    result: dict,
) -> np.ndarray:
    """
    Single annotated overlay:
      - Jet heatmap blended over smoke region
      - Green boundary contour
      - Ringelmann badge in top-left corner
    """
    out   = _alpha_heatmap_overlay(image_bgr, alpha_map, binary_mask)
    out   = _draw_boundary(out, binary_mask)
    ring  = result["ringelmann"]
    label = (f"R{ring}  α={result['representative_alpha']:.2f}"
             f"  [{result.get('quality','?')}]")
    badge_bgr = _ring_color_bgr(ring)
    tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    pad = 7
    cv2.rectangle(out, (10 - pad, 10 - pad),
                  (10 + tw + pad, 10 + th + pad), badge_bgr, -1)
    cv2.putText(out, label, (10, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def create_output_panel(
    original: np.ndarray,
    binary_mask: np.ndarray,
    trimap: np.ndarray,
    alpha_map: np.ndarray,
    result: dict,
    validation: dict,
    save_path: str | Path,
    dpi: int = 110,
) -> None:
    """
    Save a 2×3 panel:
        Row 1: Original | Smoke Mask Overlay | Trimap (colour-coded)
        Row 2: Alpha Heatmap | Alpha Overlay  | Ringelmann Gauge
    """
    rgb      = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    mask_vis = cv2.cvtColor(_draw_boundary(original, binary_mask), cv2.COLOR_BGR2RGB)
    tri_vis  = cv2.cvtColor(visualise_trimap(trimap), cv2.COLOR_BGR2RGB)
    heat_vis = cv2.cvtColor(
        _alpha_heatmap_overlay(original, alpha_map, binary_mask), cv2.COLOR_BGR2RGB
    )
    overlay_vis = cv2.cvtColor(annotate_overlay(original, binary_mask, alpha_map, result),
                                cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=dpi)

    # Row 1
    axes[0, 0].imshow(rgb);      axes[0, 0].set_title("Original",          fontsize=11)
    axes[0, 1].imshow(mask_vis); axes[0, 1].set_title("Smoke Mask (SAM3)", fontsize=11)
    axes[0, 2].imshow(tri_vis);  axes[0, 2].set_title("Trimap",            fontsize=11)
    # Trimap legend
    handles = [
        mpatches.Patch(color=(0.2, 0.2, 0.2), label="BG (0)"),
        mpatches.Patch(color=(0.0, 0.65, 1.0), label="Unknown (128)"),
        mpatches.Patch(color=(0.0, 0.78, 0.0), label="FG (255)"),
    ]
    axes[0, 2].legend(handles=handles, loc="lower right", fontsize=8)

    # Row 2
    axes[1, 0].imshow(heat_vis);
    axes[1, 0].set_title("Alpha Heatmap (jet)", fontsize=11)
    sm = ScalarMappable(cmap="jet", norm=Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Opacity α")

    axes[1, 1].imshow(overlay_vis)
    axes[1, 1].set_title("Alpha Overlay + Badge", fontsize=11)

    _ringelmann_gauge_axes(axes[1, 2], result["ringelmann"], result, validation)
    axes[1, 2].set_title("Ringelmann Result", fontsize=11)

    for ax in axes.flat[:5]:
        ax.axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)
