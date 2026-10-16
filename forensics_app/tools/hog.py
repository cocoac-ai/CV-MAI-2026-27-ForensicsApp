"""Set 4.3 - HOG (Histogram of Oriented Gradients) extraction and visualization."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image
from skimage import exposure, feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


# Set4: Core function for HOG.
# Compute HOG features and the HOG visualization image, the function takes care of image type with channel_axis
# The HOG image is drawed using cell not blocks.
def extract_hog(
    image: np.ndarray, orientations: int, pixels_per_cell: int, cells_per_block: int) -> tuple[np.ndarray, np.ndarray]:
    # Colour images have 3 channels on the last axis; grayscale has none
    channel_axis = -1 if image.ndim == 3 else None

    features, hog_image = feature.hog(
        image,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        visualize=True,
        channel_axis=channel_axis,
    )
    return features, hog_image


class HogTool(ForensicsTool):
    tool_id = "hog"
    title = "HOG features"
    category = "Set4"
    description = "Extract HOG features and show the oriented-gradient image."
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window

        params = self._ask_params(parent)
        if params is None:
            return None  # the user pressed Cancel
        orientations, pixels_per_cell, cells_per_block = params

        current = document.current
        if current.mode in ("1", "L", "LA", "I", "I;16", "F"):
            image = np.asarray(current.convert("L"))
        else:
            image = np.asarray(current.convert("RGB"))

        # The image must hold at least one block of cells
        block_size = pixels_per_cell * cells_per_block
        height, width = image.shape[:2]
        if height < block_size or width < block_size:
            return ToolResult(
                message=(
                    f"Image too small: one block needs {block_size}x{block_size} pixels. "
                    "Use smaller cells or blocks."
                )
            )

        features, hog_image = extract_hog(image, orientations, pixels_per_cell, cells_per_block)

        # HOG values are not in 0-255, so rescale them to show as an image
        hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255))
        output = Image.fromarray(hog_image.astype(np.uint8))

        return ToolResult(
            image=output,
            message=f"HOG extracted: {features.size} features.",
            details={
                "Operation": "HOG",
                "Orientations": orientations,
                "Pixels per cell": f"{pixels_per_cell} x {pixels_per_cell}",
                "Cells per block": f"{cells_per_block} x {cells_per_block}",
                "Feature vector length": features.size,
            },
        )

    def _ask_params(self, parent: tk.Misc) -> tuple[int, int, int] | None:
        window = tk.Toplevel(parent)
        window.title("HOG features")
        window.transient(parent)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)

        params_box = ttk.LabelFrame(frame, text="HOG parameters", padding=8)
        params_box.pack(fill="both")

        # (label, default value) for each parameter, defaults from the slide
        fields = [
            ("Orientations (number of angle bins):", "9"),
            ("Pixels per cell (cell = N x N pixels):", "6"),
            ("Cells per block (block = N x N cells):", "3"),
        ]
        entries = []
        for label, default in fields:
            ttk.Label(params_box, text=label).pack(anchor="w")
            entry = ttk.Entry(params_box, width=10)
            entry.pack(anchor="w", pady=(4, 8))
            entry.insert(0, default)
            entries.append(entry)

        result = None

        def apply() -> None:
            nonlocal result
            try:
                values = [int(entry.get()) for entry in entries]
            except ValueError:
                values = [0]
            if min(values) < 1:
                messagebox.showerror(
                    "Invalid value", "Please enter whole numbers of 1 or more.", parent=window
                )
                return
            result = tuple(values)
            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.pack(pady=(4, 0))
        ttk.Button(buttons, text="Apply", command=apply).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)

        window.bind("<Return>", lambda _event: apply())
        window.bind("<Escape>", lambda _event: window.destroy())
        entries[0].focus_set()
        window.grab_set()
        window.wait_window()
        return result
