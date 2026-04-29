from __future__ import annotations
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import h5py
from typing import NamedTuple

class SpotArray(NamedTuple):
    x: np.ndarray
    y: np.ndarray

    def shifted(self, dx: float, dy: float) -> "SpotArray":
        """Return the new coordinates after translation(without changing the original object)"""
        return SpotArray(self.x + dx, self.y + dy)

# ---------- Utility Function ---------- #
def diameter_px_to_s(diameter_px: float, ax: plt.Axes) -> float:
    """
    Convert the pixel diameter to the area value (pt²) of plt.scatter 
    """
    if diameter_px <= 0:
        raise ValueError("diameter_px must be positive")
    dpi = ax.figure.dpi
    diameter_pt = diameter_px * 72.0 / dpi   # px -> pt
    return diameter_pt ** 2                  # area (pt²)

def align_and_plot(
    image: np.ndarray,
    spots: SpotArray,
    dx: float,
    dy: float,
    spot_diameter_px: float = 26.0,
    title: str | None = None,
    cmap: str = "gray"
) -> SpotArray:
    """
    Manually pan the spots and draw the plot.
    The Shift SpotArray is sent back for subsequent writeback or analysis.
    """
    aligned = spots.shifted(dx, dy)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap=cmap, aspect="auto")
    s_val = diameter_px_to_s(spot_diameter_px, ax)
    ax.scatter(aligned.x, aligned.y, s=s_val, c="red", alpha=0.5)

    ax.set_axis_off()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return aligned