"""Set 5.2 - Keypoints and descriptors with ORB or SIFT (configurable).

5.3 and 5.4 import the core function extract_features from here.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from skimage import feature

from forensics_app.core import ImageDocument

from ..base import ForensicsTool, ToolResult
from .common import (
    METHOD_ORB,
    add_extractor_options,
    draw_points,
    extractor_details,
    read_extractor_options,
)


# Set5.2: Core function for descriptors.
# Detect keypoints and extract their descriptors (same calls as the slide)
def extract_features(
    gray: np.ndarray, method: str, settings: dict
) -> tuple[np.ndarray, np.ndarray]:
    if method == METHOD_ORB:
        extractor = feature.ORB(n_keypoints=settings["n_keypoints"])
    else:
        extractor = feature.SIFT(
            n_octaves=settings["n_octaves"],
            n_scales=settings["n_scales"],
            sigma_min=settings["sigma_min"],
        )
    extractor.detect_and_extract(gray)
    return extractor.keypoints, extractor.descriptors


class KeypointsTool(ForensicsTool):
    tool_id = "keypoints"
    title = "Keypoints (ORB, SIFT)"
    category = "Set 5 · SIFT"
    description = "Extract ORB or SIFT descriptors and show the keypoints."
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window

        params = self._ask_params(parent)
        if params is None:
            return None  # the user pressed Cancel
        method, settings = params

        gray = np.asarray(document.current.convert("L"))
        try:
            keypoints, descriptors = extract_features(gray, method, settings)
        except RuntimeError as exc:
            # skimage raises this when it finds no keypoints at all
            return ToolResult(message=f"{method} failed: {exc}")

        output = draw_points(document.current, keypoints)  # drawn on the colour image

        details = {"Operation": "Keypoints and descriptors"}
        details.update(extractor_details(method, settings))
        details["Keypoints found"] = len(keypoints)
        details["Descriptor length"] = descriptors.shape[1]

        return ToolResult(
            image=output,
            message=f"{method}: {len(keypoints)} keypoints extracted.",
            details=details,
        )

    def _ask_params(self, parent: tk.Misc) -> tuple[str, dict] | None:
        window = tk.Toplevel(parent)
        window.title("Keypoints (ORB, SIFT)")
        window.transient(parent)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)
        method, entries = add_extractor_options(frame)

        result = None

        def apply() -> None:
            nonlocal result
            settings = read_extractor_options(entries)
            if settings is None:
                messagebox.showerror(
                    "Invalid value",
                    "Please enter numbers above 0 (whole numbers, except sigma min).",
                    parent=window,
                )
                return
            result = (method.get(), settings)
            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.pack(pady=(12, 0))
        ttk.Button(buttons, text="Apply", command=apply).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)

        window.bind("<Return>", lambda _event: apply())
        window.bind("<Escape>", lambda _event: window.destroy())
        entries["n_keypoints"].focus_set()
        window.grab_set()
        window.wait_window()
        return result
