"""Set 2.1 - Split a colour image into its red, green and blue channels.

Could do three channels all-in-one or one-by-one.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk #For windows with theme

import numpy as np
from PIL import Image, ImageDraw, ImageFont #ImageDraw for put 3 images together

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult

CHANNEL_NAMES = ("Red", "Green", "Blue")
ALL_CHANNELS = "All"

SEPARATOR = 4
# Title font size = panel width / FONT_RATIO
# Make the titles readable for small and large images
FONT_RATIO = 10
MIN_FONT = 16


# Set2: Split RGB image, and return the R, G and B channels as 2D uint8 arrays
def split_channels(image: Image.Image) -> dict[str, np.ndarray]:
    if image.mode in {"1", "L", "LA", "I", "I;16", "F"}:
        raise ValueError("The image has a single intensity channel; there is nothing to split.")
    rgb = np.asarray(image.convert("RGB"))
    results = {}
    for index, name in enumerate(CHANNEL_NAMES):
        results[name] = rgb[:,:,index]
    return results

# Draw a canvas with titles, then put channel images together
def compose_side_by_side(panels: dict[str, Image.Image]) -> Image.Image:
    first = next(iter(panels.values())) # Get fitst pic
    width, height = first.size
    count = len(panels)

    font_size = max(MIN_FONT, width // FONT_RATIO)
    label_height = font_size * 3 // 2  # Room for the text plus a small margin
    margin = font_size // 4

    canvas = Image.new( # Use the first image to draw canvas
        first.mode,
        (width * count + SEPARATOR * (count - 1), height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(font_size)

    for index, (name, panel) in enumerate(panels.items()): # Draw three
        x = index * (width + SEPARATOR)
        canvas.paste(panel, (x, label_height))
        draw.text((x + margin, margin), name, fill="black", font=font)
    return canvas

# Show channel statistics
def channel_statistics(planes: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        f"{name} mean / std": f"{plane.mean():.1f} / {plane.std():.1f}"
        for name, plane in planes.items()
    }

# Function for separate RBG side-by-side and single channel
def split_image(image: Image.Image, channel: str) -> Image.Image:
    planes = split_channels(image)
    if channel == ALL_CHANNELS:
        panels = {}
        for name, plane in planes.items():
            panels[name] = Image.fromarray(plane)
        return compose_side_by_side(panels)
    return Image.fromarray(planes[channel])


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Channel split"
    category = "Set 2 · Image Processing"
    description = "Split the working image into red, green and blue channels."
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        try:
            planes = split_channels(document.current)
        except ValueError as error:
            return ToolResult(message=str(error), details={"Image mode": document.current.mode})

        channel = self._ask_channel(parent)
        if channel is None:
            return None

        output = split_image(document.current, channel)
        shown = "R, G, B side by side" if channel == ALL_CHANNELS else channel
        return ToolResult(
            image=output,
            message=f"Channel split: showing {shown}.",
            details={
                "Operation": "Channel split",
                "Channel": shown,
                **channel_statistics(planes),
                "Output mode": output.mode,
            },
        )

    def _ask_channel(self, parent: tk.Misc) -> str | None:
        window = tk.Toplevel(parent)
        window.title("Channel split")
        window.transient(parent)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)

        channel = tk.StringVar(value=ALL_CHANNELS)

        channel_box = ttk.LabelFrame(frame, text="Channel", padding=8)
        channel_box.pack(fill="both")
        ttk.Radiobutton(
            channel_box, text="All (side by side)", value=ALL_CHANNELS, variable=channel,
        ).pack(anchor="w")
        for name in CHANNEL_NAMES:
            ttk.Radiobutton(channel_box, text=name, value=name, variable=channel).pack(anchor="w")

        result: str | None = None

        def apply() -> None:
            nonlocal result
            result = channel.get()
            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.pack(pady=(12, 0))
        ttk.Button(buttons, text="Apply", command=apply).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)

        window.bind("<Return>", lambda _event: apply())
        window.bind("<Escape>", lambda _event: window.destroy())
        window.grab_set()
        window.wait_window()
        return result
