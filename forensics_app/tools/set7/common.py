"""Set7 - Code shared by the Set 7 tools (7.1 codebook, 7.2 histogram, 7.3 classification).

7.1 saves everything it computes (the trained SVM has its own file); 7.2 and 7.3 load it.
"""

from __future__ import annotations

import json
import pickle
import tkinter as tk
from io import BytesIO
from pathlib import Path
from tkinter import ttk
from typing import Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch
from PIL import Image

DATASET_FOLDER = Path("images/set7-images")
MODEL_FILE = Path("models/bow_model.npz")
CLASSIFIER_FILE = Path("models/bow_classifier.pkl")  # an SVM cannot go in a .npz file
NO_MODEL_MESSAGE = "No codebook found. Run Set 7.1 (Codebook) first."

# SIFT can find 2000 keypoints in one image: drawing them all would hide the image and the
# histogram, so only this many are drawn (the histogram itself still counts all of them)
MAX_DRAWN = 40
PREVIEW_DPI = 200  # the workspace is bigger than the window, so the text stays sharp


# ---------------------------------------------------------------------------
# Model file (7.2, 7.3)
# ---------------------------------------------------------------------------

# Read the files saved by 7.1; None if they do not exist yet
def load_model() -> dict | None:
    if not MODEL_FILE.is_file() or not CLASSIFIER_FILE.is_file():
        return None
    with np.load(MODEL_FILE) as data:
        model = dict(data)
    model["classifier"] = pickle.loads(CLASSIFIER_FILE.read_bytes())
    return model


def model_method(model: dict) -> tuple[str, dict]:
    return str(model["method"]), json.loads(str(model["settings"]))


# Keypoints and descriptors of one dataset image.
# All the descriptors are stacked in one array; "owner" says which image each row came from.
def image_features(model: dict, index: int) -> tuple[np.ndarray, np.ndarray]:
    rows = model["owner"] == index
    return model["keypoints"][rows], model["descriptors"][rows]


# ---------------------------------------------------------------------------
# Windows and drawing (7.1, 7.2, 7.3)
# ---------------------------------------------------------------------------

# Non-modal window with a Matplotlib figure, like the Set 2.4 histogram.
# With on_classify, the window also has a "Classify" button (7.3).
def show_figure(
    parent: tk.Misc, figure: Figure, title: str, on_classify: Callable[[], None] | None = None
) -> None:
    window = tk.Toplevel(parent)
    window.title(title)

    canvas = FigureCanvasTkAgg(figure, master=window)
    NavigationToolbar2Tk(canvas, window).update()
    if on_classify is not None:
        ttk.Button(window, text="Classify", command=on_classify).pack(side="bottom", pady=8)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw_idle()


# The figure as an image, to show it in the workspace too (ToolResult.preview)
def figure_image(figure: Figure) -> Image.Image:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=PREVIEW_DPI)
    return Image.open(buffer).convert("RGB")


# One row of the figure, like the slide: the image on the left, its histogram on the right and
# lines from some keypoints to the bar of their visual word. Returns the axes of the image.
def draw_image_and_histogram(
    figure: Figure,
    row: int,
    rows: int,
    image: Image.Image,
    keypoints: np.ndarray,
    words: np.ndarray,
    histogram: np.ndarray,
    title: str,
) -> Axes:
    chosen = np.linspace(0, len(words) - 1, min(MAX_DRAWN, len(words))).astype(int)

    image_axes = figure.add_subplot(rows, 2, 2 * row + 1)
    image_axes.imshow(image)
    rows_drawn, cols_drawn = keypoints[chosen].T
    image_axes.scatter(cols_drawn, rows_drawn, c=words[chosen], cmap="tab20", s=8)
    image_axes.set_title(title)
    image_axes.axis("off")

    histogram_axes = figure.add_subplot(rows, 2, 2 * row + 2)
    histogram_axes.bar(
        np.arange(len(histogram)), histogram, color="lightgray", edgecolor="black", linewidth=0.5
    )
    histogram_axes.set_title("Histogram of visual words")
    histogram_axes.set_xlabel("Visual word index")
    histogram_axes.set_ylabel("Normalized frequency")

    for index in chosen:
        row_pixel, col_pixel = keypoints[index]
        word = words[index]
        figure.add_artist(
            ConnectionPatch(
                xyA=(col_pixel, row_pixel),
                coordsA="data",
                axesA=image_axes,
                xyB=(word, histogram[word]),
                coordsB="data",
                axesB=histogram_axes,
                color="gray",
                linewidth=0.3,
                alpha=0.5,
            )
        )
    return image_axes
