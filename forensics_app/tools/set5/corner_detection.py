"""Set 5.1 - Corner detection with FAST or Harris and a modifiable peak threshold."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib import cm
from PIL import Image
from skimage import exposure, feature, img_as_ubyte

from forensics_app.core import ImageDocument

from ..base import ForensicsTool, ToolResult
from ..common import compose_side_by_side
from .common import draw_points

METHOD_FAST = "FAST"
METHOD_HARRIS = "Harris"


# Set5.1: Core function for corners.
# Corner response image (FAST or Harris): how much each pixel looks like a corner
def corner_response(gray: np.ndarray, method: str, settings: dict) -> np.ndarray:
    if method == METHOD_FAST:
        return feature.corner_fast(
            gray, n=settings["fast_n"], threshold=settings["fast_threshold"]
        )
    return feature.corner_harris(
        gray, k=settings["harris_k"], sigma=settings["harris_sigma"]
    )


# Keep the peaks of the response image as corner points; each row is (row, col) of one corner
def find_peaks(response: np.ndarray, settings: dict) -> np.ndarray:
    return feature.corner_peaks(
        response, min_distance=settings["min_distance"], threshold_rel=settings["threshold_rel"]
    )


# Set5.1: Response image, then its peaks
def detect_corners(gray: np.ndarray, method: str, settings: dict) -> np.ndarray:
    return find_peaks(corner_response(gray, method, settings), settings)


# Show the response image with a colour map (like plt.imshow in the slide)
def response_image(response: np.ndarray) -> Image.Image:
    scaled = exposure.rescale_intensity(response.astype(float), out_range=(0, 1))
    rgb = cm.viridis(scaled)[..., :3]  # drop the alpha channel
    return Image.fromarray(img_as_ubyte(rgb))


class CornerDetectionTool(ForensicsTool):
    tool_id = "corner_detection"
    title = "Corner detection"
    category = "Set 5 · SIFT"
    description = "Find corners with FAST or Harris and mark them in red."
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window

        params = self._ask_params(parent)
        if params is None:
            return None  # the user pressed Cancel
        method, settings = params

        # Corners are found on the grayscale image, then drawn on the colour one
        gray = np.asarray(document.current.convert("L"))
        response = corner_response(gray, method, settings)
        points = find_peaks(response, settings)
        output = compose_side_by_side({
            f"{method} response": response_image(response),
            "Corners": draw_points(document.current, points),
        })

        details = {"Operation": "Corner detection", "Method": method}
        if method == METHOD_FAST:
            details["Pixels in a row (n)"] = settings["fast_n"]
            details["Brightness threshold"] = settings["fast_threshold"]
        else:
            details["k"] = settings["harris_k"]
            details["Sigma"] = settings["harris_sigma"]
        details["Peak threshold (relative)"] = settings["threshold_rel"]
        details["Min distance between corners"] = settings["min_distance"]
        details["Corners found"] = len(points)

        return ToolResult(
            image=output,
            message=f"{method}: {len(points)} corners found.",
            details=details,
        )

    def _ask_params(self, parent: tk.Misc) -> tuple[str, dict] | None:
        window = tk.Toplevel(parent)
        window.title("Corner detection")
        window.transient(parent)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)

        # Detector
        method_box = ttk.LabelFrame(frame, text="Corner detector", padding=8)
        method_box.pack(fill="both")
        method = tk.StringVar(value=METHOD_FAST)
        for name in (METHOD_FAST, METHOD_HARRIS):
            ttk.Radiobutton(method_box, text=name, value=name, variable=method).pack(anchor="w")

        # (key, label, default) for each parameter box. FAST and Harris use the skimage defaults;
        # the peak defaults are our own choice (skimage: no threshold, min_distance=1),
        # because without a threshold Harris marks thousands of weak points.
        fast_fields = [
            ("fast_n", "Pixels in a row on the circle, 1 to 16 (n):", "12"),
            ("fast_threshold", "Brightness threshold, 0 to 1:", "0.15"),
        ]
        harris_fields = [
            ("harris_k", "k, 0 to 0.25 (higher = fewer corners):", "0.05"),
            ("harris_sigma", "Sigma (size of the area around each pixel):", "1"),
        ]
        peaks_fields = [
            ("threshold_rel", "Threshold, 0 to 1 (higher = fewer corners):", "0.1"),
            ("min_distance", "Min distance between corners in pixels:", "5"),
        ]
        entries = {}
        boxes = (
            ("FAST parameters", fast_fields),
            ("Harris parameters", harris_fields),
            ("Peaks", peaks_fields),
        )
        for box_title, fields in boxes:
            box = ttk.LabelFrame(frame, text=box_title, padding=8)
            box.pack(fill="both", pady=(8, 0))
            for key, label, default in fields:
                ttk.Label(box, text=label).pack(anchor="w")
                entry = ttk.Entry(box, width=10)
                entry.pack(anchor="w", pady=(4, 8))
                entry.insert(0, default)
                entries[key] = entry

        result = None

        def apply() -> None:
            nonlocal result
            try:
                settings = {
                    "fast_n": int(entries["fast_n"].get()),
                    "fast_threshold": float(entries["fast_threshold"].get()),
                    "harris_k": float(entries["harris_k"].get()),
                    "harris_sigma": float(entries["harris_sigma"].get()),
                    "threshold_rel": float(entries["threshold_rel"].get()),
                    "min_distance": int(entries["min_distance"].get()),
                }
            except ValueError:
                settings = None
            if (
                settings is None
                or not 1 <= settings["fast_n"] <= 16
                or not 0 <= settings["fast_threshold"] <= 1
                or not 0 < settings["harris_k"] < 0.25
                or settings["harris_sigma"] <= 0
                or not 0 <= settings["threshold_rel"] <= 1
                or settings["min_distance"] < 1
            ):
                messagebox.showerror(
                    "Invalid value",
                    "Please check that every value is a number in the range shown.",
                    parent=window,
                )
                return
            result = (method.get(), settings)
            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.pack(pady=(4, 0))
        ttk.Button(buttons, text="Apply", command=apply).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)

        window.bind("<Return>", lambda _event: apply())
        window.bind("<Escape>", lambda _event: window.destroy())
        entries["threshold_rel"].focus_set()
        window.grab_set()
        window.wait_window()
        return result
