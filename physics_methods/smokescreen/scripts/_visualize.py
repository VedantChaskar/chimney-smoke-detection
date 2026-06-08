"""
scripts/_visualize.py

Shared visualization helpers for smokescreen scripts.
Not meant to be called directly.
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Ringelmann colour palette (Grade 0 = white, Grade 5 = black)
_GRADE_COLOURS = {
    0: "#FFFFFF",
    1: "#CCCCCC",
    2: "#999999",
    3: "#666666",
    4: "#333333",
    5: "#000000",
}

_GRADE_TEXT_COLOURS = {
    0: "black", 1: "black", 2: "black",
    3: "white", 4: "white", 5: "white",
}


def _overlay_regions(img_bgr: np.ndarray, regions_summary: dict) -> np.ndarray:
    """Return a copy of the image with region info overlaid as text."""
    out = img_bgr.copy()
    y   = 20
    for name, info in regions_summary.items():
        if info is None:
            txt = f"{name}: (not detected)"
        else:
            txt = (f"{name}: brightness={info['brightness']:.0f} "
                   f"area={info['area']} conf={info['confidence']:.2f}")
        cv2.putText(out, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return out


def save_panel(image_bgr: np.ndarray, result: dict, out_path: Path) -> None:
    """
    Save a 2×2 matplotlib figure showing:
      [0,0] Original image
      [0,1] Region info overlay
      [1,0] Opacity bar chart (L_smoke vs L_bright vs L_dark)
      [1,1] Ringelmann grade badge
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(Path(out_path.stem.replace("_smokescreen", "")).name, fontsize=13)

    # Panel 0: original
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    # Panel 1: region overlay
    overlay = _overlay_regions(image_bgr, result.get("regions_summary", {}))
    axes[0, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Detected Regions")
    axes[0, 1].axis("off")

    # Panel 2: brightness bar chart
    ax = axes[1, 0]
    labels = ["L_dark", "L_smoke", "L_bright"]
    values = [
        result.get("L_dark")   or 0,
        result.get("L_smoke")  or 0,
        result.get("L_bright") or 0,
    ]
    colours = ["#444444", "#888888", "#CCCCCC"]
    bars = ax.bar(labels, values, color=colours, edgecolor="black")
    ax.set_ylim(0, 260)
    ax.set_ylabel("Brightness (0–255)")
    ax.set_title(f"Brightness References  [{result.get('formula', 'N/A')}]")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{val:.0f}", ha="center", fontsize=10)
    ax.axhline(y=result.get("L_smoke") or 0, color="red", linestyle="--",
               linewidth=0.8, label="L_smoke")
    ax.legend(fontsize=9)

    # Panel 3: Ringelmann grade badge
    ax = axes[1, 1]
    grade  = result.get("grade", 0)
    opacity = result.get("opacity", 0.0)
    label   = result.get("label", f"Grade {grade}")
    bg_col  = _GRADE_COLOURS.get(grade, "#888888")
    txt_col = _GRADE_TEXT_COLOURS.get(grade, "black")

    ax.set_facecolor(bg_col)
    ax.text(0.5, 0.6, f"R{grade}", ha="center", va="center",
            fontsize=64, color=txt_col, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha="center", va="center",
            fontsize=12, color=txt_col, transform=ax.transAxes)
    ax.text(0.5, 0.12, f"Opacity: {opacity:.1f}%", ha="center", va="center",
            fontsize=10, color=txt_col, transform=ax.transAxes)
    reliable_str = "✓ reliable" if result.get("reliable") else "⚠ proxy refs"
    ax.text(0.5, 0.04, reliable_str, ha="center", va="center",
            fontsize=9, color=txt_col, alpha=0.8, transform=ax.transAxes)
    ax.set_title("Ringelmann Grade", color="black")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
