"""Set 2.4 - Intensity histogram of the working image.
The histogram opens in its own window, so the working image is never replaced.
"""

from __future__ import annotations

import tkinter as tk

from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


LEVELS = 256

CHANNEL_COLOURS = {
    "Grayscale": "black",
    "Red": "tab:red",
    "Green": "tab:green",
    "Blue": "tab:blue",
}


# Normalise any PIL mode to 8-bit L or RGB.
def to_8bit(image: Image.Image) -> Image.Image:
    if image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")

    if image.mode.startswith("I;16"):
        return Image.fromarray((np.asarray(image, dtype=np.uint16) >> 8).astype(np.uint8))

    if image.mode in {"1", "L", "LA", "I", "F"}:
        return image.convert("L")

    return image.convert("RGB")


# Pixel counts per intensity level (0-255).
def compute_histograms(image: Image.Image) -> dict[str, np.ndarray]:
    image = to_8bit(image)

    if image.mode == "RGB":
        rgb = np.asarray(image)
        results = {}
        for index, name in enumerate(("Red", "Green", "Blue")):
            results[name] = freq_counts(rgb[:, :, index])
        return results
    
    return {"Grayscale": freq_counts(np.asarray(image.convert("L")))}

# Set3: Calculate the frequency of each intensity level using NumPy
def freq_counts(plane: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(plane.ravel(),bins=LEVELS,range=(0, LEVELS))
    return counts

# Summary statistics derived from a 256-bin histogram."""
def histogram_statistics(counts: np.ndarray) -> dict[str, float]:
    total = int(counts.sum())
    levels = np.arange(LEVELS)
    occupied = np.flatnonzero(counts)

    mean = float((levels * counts).sum() / total)
    cumulative = np.cumsum(counts)

    return {
        "mean": mean,
        "median": int(np.searchsorted(cumulative, total / 2)),
        "std": float(np.sqrt((counts * (levels - mean) ** 2).sum() / total)),
        "min": int(occupied[0]),
        "max": int(occupied[-1]),
        "clipped_low": 100.0 * counts[0] / total,
        "clipped_high": 100.0 * counts[-1] / total,
    }

def draw_histograms(axes: Axes, histograms: dict[str, np.ndarray]) -> None:
    axes.clear()
    levels = np.arange(LEVELS)

    for name, counts in histograms.items():
        colour = CHANNEL_COLOURS[name]
        axes.plot(levels,counts, color=colour, linewidth=1, label=name)

    axes.set_xlim(0, LEVELS - 1)
    axes.set_ylim(bottom=0)
    axes.set_xlabel("Pixel intensity")
    axes.set_ylabel("Frequency")
    axes.grid(alpha=0.3)

    if len(axes.get_lines()) > 1:
        axes.legend(loc="upper right", fontsize="small")

# Non-modal window with an interactive Matplotlib histogram.
class HistogramWindow(tk.Toplevel):
    def __init__(self, parent: tk.Misc, histograms: dict[str, np.ndarray],title: str) -> None:
        super().__init__(parent)

        self.title(f"Histogram — {title}")
        self.histograms = histograms

        self.figure = Figure(figsize=(6.4, 3.6), dpi=100, layout="tight")
        self.axes = self.figure.add_subplot()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)

        NavigationToolbar2Tk(self.canvas, self).update()

        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.redraw()

    # Redraw the histogram when the scale changes."""
    def redraw(self) -> None:
        draw_histograms(self.axes, self.histograms)
        self.canvas.draw_idle()


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Histogram"
    category = "Image"
    description = "Show grayscale or RGB channel histograms in a separate window."
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None

        histograms = compute_histograms(document.current)
        name = document.path.name if document.path else "working image"

        self._open_window(parent, histograms, name)

        details: dict[str, object] = {"Operation": "Histogram", "Channels": ", ".join(histograms),}

        for channel, counts in histograms.items():
            statistics = histogram_statistics(counts)

            details[f"{channel} mean"] = (f"{statistics['mean']:.1f}")
            details[f"{channel} median"] = statistics["median"]
            details[f"{channel} std"] = (f"{statistics['std']:.1f}")
            details[f"{channel} range"] = (f"{statistics['min']} – {statistics['max']}")
            details[f"{channel} clipped at 0"] = (f"{statistics['clipped_low']:.2f}%")
            details[f"{channel} clipped at 255"] = (f"{statistics['clipped_high']:.2f}%")

        return ToolResult(
            message=("Histogram opened in a new window; the working image is unchanged."),
            details=details,
            )

    def _open_window(self, parent: tk.Misc, histograms: dict[str, np.ndarray], name: str) -> None:
        HistogramWindow(parent, histograms, name)