"""Set 3.5 - Image sharpening by unsharp masking."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image
from skimage import filters, util

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


RADIUS_RANGE = (0.1, 20.0)
AMOUNT_RANGE = (0.0, 10.0)


@dataclass(frozen=True)
class SharpeningParams:
    radius: float = 2.0
    amount: float = 1.5


@dataclass(frozen=True)
class UnsharpMaskResult:
    blurred: np.ndarray
    detail: np.ndarray
    sharpened: np.ndarray

# Set3: Sharpen a float image in [0, 1] using unsharp masking.
def unsharp_mask(image: np.ndarray, radius: float, amount: float) -> UnsharpMaskResult:
    if radius <= 0:
        raise ValueError("radius must be positive")
    if amount < 0:
        raise ValueError("amount must be non-negative")

    # Identify the color channel dimension for RGB images.
    if image.ndim == 3:
        channel_axis = -1
    else:
        channel_axis = None

    # Apply Gaussian blur (same call as the slide)
    blurred = filters.gaussian(image, sigma=radius, channel_axis=channel_axis)

    # Extract details from the original image.
    detail = image - blurred

    # Add enhanced details back to the original image, constrain range to 0,1
    sharpened = np.clip(image + amount * detail, 0.0, 1.0)

    return UnsharpMaskResult(blurred=blurred, detail=detail, sharpened=sharpened)

# Apply unsharp masking; grayscale stays grayscale, everything else becomes RGB
def sharpen_image(image: Image.Image, params: SharpeningParams) -> Image.Image:
    if image.mode in {"1", "L", "LA", "I", "I;16", "F"}:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    array = util.img_as_float(np.asarray(image))

    result = unsharp_mask(array, params.radius, params.amount)

    return Image.fromarray(util.img_as_ubyte(result.sharpened))


class SharpeningTool(ForensicsTool):
    tool_id = "sharpening"
    title = "Image sharpening"
    category = "Set3"
    description = (
        "Sharpen the image with unsharp masking "
        "(configurable radius and amount)."
    )
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        params = self._ask_params(parent)

        if params is None:
            return None

        output = sharpen_image(document.current, params)

        return ToolResult(
            image=output,
            message=(
                f"Unsharp masking applied "
                f"(radius={params.radius:g}, amount={params.amount:g})."
            ),
            details={
                "Operation": "Unsharp masking",
                "Radius (sigma)": params.radius,
                "Amount": params.amount,
                "Output mode": output.mode,
            },
        )

    def _ask_params(self, parent: tk.Misc) -> SharpeningParams | None:
        defaults = SharpeningParams()

        window = tk.Toplevel(parent)
        window.title("Image sharpening")
        window.transient(parent)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)

        radius = tk.DoubleVar(value=defaults.radius)
        amount = tk.DoubleVar(value=defaults.amount)

        params_box = ttk.LabelFrame(frame, text="Unsharp mask", padding=8)
        params_box.pack(fill="both")

        ttk.Label(params_box, text=f"Radius / Gaussian sigma ({RADIUS_RANGE[0]:g}-{RADIUS_RANGE[1]:g}):").pack(anchor="w")
        radius_entry = ttk.Entry(params_box, textvariable=radius, width=10)
        radius_entry.pack(anchor="w", pady=(4, 8))

        ttk.Label(params_box, text=f"Amount ({AMOUNT_RANGE[0]:g}-{AMOUNT_RANGE[1]:g}):").pack(anchor="w")
        ttk.Entry(params_box, textvariable=amount, width=10).pack(anchor="w", pady=(4, 0))

        result: SharpeningParams | None = None

        def apply() -> None:
            nonlocal result

            try:
                r = radius.get()
                a = amount.get()
            except tk.TclError:
                messagebox.showerror(
                    "Invalid value",
                    "Please enter numbers only.",
                    parent=window,
                )
                return

            if not RADIUS_RANGE[0] <= r <= RADIUS_RANGE[1]:
                messagebox.showerror(
                    "Invalid radius",
                    (
                        f"Radius must be between "
                        f"{RADIUS_RANGE[0]:g} and "
                        f"{RADIUS_RANGE[1]:g}."
                    ),
                    parent=window,
                )
                return

            if not AMOUNT_RANGE[0] <= a <= AMOUNT_RANGE[1]:
                messagebox.showerror(
                    "Invalid amount",
                    (
                        f"Amount must be between "
                        f"{AMOUNT_RANGE[0]:g} and "
                        f"{AMOUNT_RANGE[1]:g}."
                    ),
                    parent=window,
                )
                return

            result = SharpeningParams(
                radius=r,
                amount=a,
            )

            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.pack(pady=(12, 0))
        ttk.Button(buttons, text="Apply", command=apply).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)

        window.bind("<Return>", lambda _event: apply())
        window.bind("<Escape>", lambda _event: window.destroy())
        radius_entry.focus_set()
        window.grab_set()
        window.wait_window()

        return result