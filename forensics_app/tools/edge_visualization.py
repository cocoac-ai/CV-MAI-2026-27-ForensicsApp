"""Set 3.4 - Show Canny edges superimposed on the original image."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from skimage import color, feature, img_as_float, img_as_ubyte

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

# The four visualizations from the Set 3.4 slide.
STYLE_RED = "Red edges"
STYLE_TRANSPARENT = "Transparent overlay"
STYLE_DIMMED = "Bright edges, dim image"
STYLE_CONTOUR = "Contour overlay"
STYLES = (STYLE_RED, STYLE_TRANSPARENT, STYLE_DIMMED, STYLE_CONTOUR)

# How strongly the red overlay is mixed in (like imshow(overlay, alpha=0.5)).
OVERLAY_ALPHA = 0.5

# Set3: Canny edge map (True = edge) of the image's grayscale version
def detect_edges(image: Image.Image, sigma: float) -> np.ndarray:
    gray = img_as_float(np.asarray(image.convert("L")))
    return feature.canny(gray, sigma=sigma)

# Draw edges on a grayscale copy of image and return an RGB image.
def overlay_edges(image: Image.Image, edges: np.ndarray, style: str) -> Image.Image:
    # Grayscale background, so the red edges stand out
    gray = img_as_float(np.asarray(image.convert("L")))
    rgb = color.gray2rgb(gray)

    # 1. Red edges on grayscale image
    if style == STYLE_RED:
        result = rgb.copy()
        result[edges] = [1, 0, 0]
    # 2. Transparent red overlay: a black image with red edges,
    elif style == STYLE_TRANSPARENT:
        overlay = np.zeros_like(rgb)
        overlay[..., 0] = edges.astype(float)
        # For adding the layers together
        result = (1 - OVERLAY_ALPHA) * rgb + OVERLAY_ALPHA * overlay
    elif style == STYLE_DIMMED:
    # 3. Darken everything except edges
        result = rgb * 0.4
        result[edges] = [1, 1, 1]
    elif style == STYLE_CONTOUR:
    # 4. Show edges as contours
        result = draw_contours(gray, edges)
    else:
        raise ValueError(f"Unknown style: {style}")

    return Image.fromarray(img_as_ubyte(result))

# For option 4 to draw into an image
def draw_contours(gray: np.ndarray, edges: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    # Create canvas
    figure = Figure(figsize=(width / 100, height / 100), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))

    ax.imshow(gray, cmap="gray")
    ax.contour(edges, colors="red", linewidths=0.8)
    ax.axis("off")

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba())[..., :3]  # drop the alpha channel
    return img_as_float(pixels)


class EdgeVisualizationTool(ForensicsTool):
    tool_id = "edge_visualization"
    title = "Edge visualization"
    category = "Set3"
    description = "Detect edges with Canny and draw them on top of the image."
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window

        params = self._ask_params(parent)
        if params is None:
            return None  # the user pressed Cancel
        sigma, style = params

        edges = detect_edges(document.current, sigma)
        output = overlay_edges(document.current, edges, style)

        edge_percent = 100.0 * edges.sum() / edges.size
        return ToolResult(
            image=output,
            message=f"Edges superimposed ({style}, sigma={sigma:g}).",
            details={
                "Operation": "Edge visualization",
                "Edge detector": "Canny",
                "Sigma": sigma,
                "Style": style,
                "Edge pixels": f"{int(edges.sum())} ({edge_percent:.2f}%)",
            },
        )

    def _ask_params(self, parent: tk.Misc) -> tuple[float, str] | None:
        """Small dialog asking for sigma and the overlay style."""
        window = tk.Toplevel(parent)
        window.title("Edge visualization")
        window.transient(parent)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)

        sigma_box = ttk.LabelFrame(frame, text="Canny", padding=8)
        sigma_box.pack(fill="both")
        ttk.Label(sigma_box, text="Sigma (larger = fewer, smoother edges):").pack(anchor="w")
        sigma_entry = ttk.Entry(sigma_box, width=10)
        sigma_entry.pack(anchor="w", pady=(4, 0))
        sigma_entry.insert(0, "2")

        style_box = ttk.LabelFrame(frame, text="Style", padding=8)
        style_box.pack(fill="both", pady=(8, 0))
        style = tk.StringVar(value=STYLE_RED)
        for name in STYLES:
            ttk.Radiobutton(style_box, text=name, value=name, variable=style).pack(anchor="w")

        result = None

        def apply() -> None:
            nonlocal result
            try:
                sigma = float(sigma_entry.get())
            except ValueError:
                sigma = -1.0
            if sigma <= 0:
                messagebox.showerror("Invalid sigma", "Sigma must be a positive number.", parent=window)
                return
            result = (sigma, style.get())
            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.pack(pady=(12, 0))
        ttk.Button(buttons, text="Apply", command=apply).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)

        window.bind("<Return>", lambda _event: apply())
        window.bind("<Escape>", lambda _event: window.destroy())
        sigma_entry.focus_set()
        window.grab_set()
        window.wait_window()
        return result
