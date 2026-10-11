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

    # Apply Gaussian blur, reflect for edges
    blurred = filters.gaussian(image, sigma=radius, mode="reflect",channel_axis=channel_axis)

    # Extract details from the original image.
    detail = image - blurred

    # Add enhanced details back to the original image, constrain range to 0,1
    sharpened = np.clip(image + amount * detail, 0.0, 1.0)

    return UnsharpMaskResult(blurred=blurred, detail=detail, sharpened=sharpened)

# Return an L/RGB base image and the alpha channel, if present
def split_alpha(image: Image.Image) -> tuple[Image.Image, Image.Image | None]:

    if image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")

    alpha = image.getchannel("A") if "A" in image.getbands() else None

    gray_modes = {"1", "L", "LA", "I", "I;16", "F"}

    if image.mode in gray_modes:
        base = image.convert("L")
    else:
        base = image.convert("RGB")

    return base, alpha

# Apply unsharp masking while preserving color and transparency
def sharpen_image(image: Image.Image, params: SharpeningParams) -> Image.Image:
    base, alpha = split_alpha(image)
    array = util.img_as_float(np.asarray(base))

    result = unsharp_mask(array, params.radius, params.amount)

    sharpened = Image.fromarray(util.img_as_ubyte(result.sharpened))

    if alpha is not None:
        sharpened.putalpha(alpha)

    return sharpened


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

        ttk.Label(
            frame,
            text="Radius (Gaussian sigma):",
        ).grid(row=0, column=0, sticky="w")

        ttk.Spinbox(
            frame,
            from_=RADIUS_RANGE[0],
            to=RADIUS_RANGE[1],
            increment=0.5,
            textvariable=radius,
            width=8,
        ).grid(
            row=0,
            column=1,
            padx=(8, 0),
            pady=4,
        )

        ttk.Label(
            frame,
            text="Amount:",
        ).grid(
            row=1,
            column=0,
            sticky="w",
        )

        ttk.Spinbox(
            frame,
            from_=AMOUNT_RANGE[0],
            to=AMOUNT_RANGE[1],
            increment=0.5,
            textvariable=amount,
            width=8,
        ).grid(
            row=1,
            column=1,
            padx=(8, 0),
            pady=4,
        )

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
        buttons.grid(
            row=2,
            column=0,
            columnspan=2,
            pady=(12, 0),
        )

        ttk.Button(
            buttons,
            text="Apply",
            command=apply,
        ).pack(side="left", padx=4)

        ttk.Button(
            buttons,
            text="Cancel",
            command=window.destroy,
        ).pack(side="left", padx=4)

        window.bind("<Return>", lambda _event: apply())
        window.bind("<Escape>", lambda _event: window.destroy())

        window.grab_set()
        window.wait_window()

        return result