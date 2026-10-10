"""Set 2.5 - Contrast stretching with percentiles.

For colour images the same two percentiles are used for R, G and B, so the
colour balance of the image is kept.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import exposure

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

# Stretch a uint8 image (gray or RGB). Returns (result, p_low, p_high)."""
def stretch_contrast(image: np.ndarray, clip_percent: float) -> tuple[np.ndarray, float, float]:
    # One pair of percentiles computed over all channels together.
    p_low, p_high = np.percentile(image, (clip_percent, 100 - clip_percent))

    # A flat image has nothing to stretch (and would divide by zero).
    if p_high <= p_low:
        return image.copy(), float(p_low), float(p_high)

    stretched = exposure.rescale_intensity(
        image,
        in_range=(p_low, p_high),
        out_range=(0, 255),
    )
    stretched = np.round(stretched).astype(np.uint8)
    return stretched, float(p_low), float(p_high)


class ContrastStretchingTool(ForensicsTool):
    tool_id = "contrast_stretching"
    title = "Contrast stretching"
    category = "Set2"
    description = "Stretch the intensity range between two percentiles to 0-255."
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window

        clip_percent = simpledialog.askfloat(
            "Contrast stretching",
            "Percentage of pixels to clip at each end (0-49):",
            initialvalue=2.0,
            minvalue=0.0,
            maxvalue=49.0,
            parent=parent,
        )
        if clip_percent is None:
            return None  # the user pressed Cancel

        image = document.current

        # Keep the alpha (transparency) channel aside; it is not stretched.
        alpha = image.getchannel("A") if "A" in image.getbands() else None

        # Work on 8-bit grayscale or 8-bit RGB.
        if image.mode in {"1", "L", "LA", "I", "I;16", "F"}:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        stretched, p_low, p_high = stretch_contrast(np.asarray(image), clip_percent)

        output = Image.fromarray(stretched)
        if alpha is not None:
            output.putalpha(alpha)

        return ToolResult(
            image=output,
            message=f"Contrast stretching applied ({clip_percent:g}% clipped at each end).",
            details={
                "Operation": "Contrast stretching",
                "Clip percent": clip_percent,
                f"Low percentile ({clip_percent:g}%)": f"{p_low:.1f}",
                f"High percentile ({100 - clip_percent:g}%)": f"{p_high:.1f}",
                "Mapping": f"[{p_low:.0f}, {p_high:.0f}] -> [0, 255]",
                "Output mode": output.mode,
            },
        )
