"""Set 2.1 - Split a colour image into its red, green and blue channels."""

from __future__ import annotations

from dataclasses import dataclass #For data structure
import tkinter as tk
from tkinter import ttk #For windows with theme

import numpy as np
from PIL import Image, ImageDraw, ImageFont #ImageDraw for titles...

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

CHANNEL_NAMES = ("Red", "Green", "Blue")
ALL_CHANNELS = "All"

DISPLAY_GRAY = "gray"
DISPLAY_TINTED = "tinted"

LABEL_HEIGHT = 50
SEPARATOR = 4
IMAGE_FONT = 25


@dataclass(frozen=True) #No need for initialization if use dataclass
class ChannelSplitParams:
    channel: str = ALL_CHANNELS
    display: str = DISPLAY_GRAY

#Split RGB or RGBA image, and return the R, G and B channels as 2D uint8 arrays
def split_channels(image: Image.Image) -> dict[str, np.ndarray]:
    if image.mode in {"1", "L", "LA", "I", "I;16", "F"}:
        raise ValueError("The image has a single intensity channel; there is nothing to split.")
    rgb = np.asarray(image.convert("RGB"))
    results = {}
    for index, name in enumerate(CHANNEL_NAMES):
        results[name] = rgb[:,:,index]
    return results

#Show a channel as grayscale, or placed back into its own RGB slot
def render_channel(plane: np.ndarray, name: str, display: str) -> Image.Image:
    if display == DISPLAY_GRAY:
        return Image.fromarray(plane)
    height, width = plane.shape
    tinted = np.zeros((height, width, 3), dtype=np.uint8)
    tinted[:, :, CHANNEL_NAMES.index(name)] = plane
    return Image.fromarray(tinted)

#Draw a canvas with titles, then put channel images together
def compose_side_by_side(panels: dict[str, Image.Image]) -> Image.Image:
    first = next(iter(panels.values())) #Get fitst pic
    width, height = first.size
    count = len(panels)
    canvas = Image.new(
        first.mode,
        (width * count + SEPARATOR * (count - 1), height + LABEL_HEIGHT),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(IMAGE_FONT)
    for index, (name, panel) in enumerate(panels.items()):
        x = index * (width + SEPARATOR)
        canvas.paste(panel, (x, LABEL_HEIGHT))
        draw.text((x + 4, 3), name, fill="black", font=font)
    return canvas


def channel_statistics(planes: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        f"{name} mean / std": f"{plane.mean():.1f} / {plane.std():.1f}"
        for name, plane in planes.items()
    }


def split_image(image: Image.Image, params: ChannelSplitParams) -> Image.Image:
    planes = split_channels(image)
    if params.channel == ALL_CHANNELS:
        panels = {}
        for name, plane in planes.items():
            panels[name] = render_channel(plane, name, params.display)
        return compose_side_by_side(panels)
    return render_channel(planes[params.channel], params.channel, params.display)


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Channel split"
    category = "Set2"
    description = "Split the working image into red, green and blue channels."
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window

        try:
            planes = split_channels(document.current)
        except ValueError as error:
            return ToolResult(message=str(error), details={"Image mode": document.current.mode})

        params = self._ask_params(parent)
        if params is None:
            return None

        output = split_image(document.current, params)
        shown = "R, G, B side by side" if params.channel == ALL_CHANNELS else params.channel
        return ToolResult(
            image=output,
            message=f"Channel split: showing {shown} ({params.display}).",
            details={
                "Operation": "Channel split",
                "Channel": shown,
                "Display": params.display,
                **channel_statistics(planes),
                "Output mode": output.mode,
            },
        )

    def _ask_params(self, parent: tk.Misc) -> ChannelSplitParams | None:
        defaults = ChannelSplitParams()
        window = tk.Toplevel(parent)
        window.title("Channel split")
        window.transient(parent)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)

        channel = tk.StringVar(value=defaults.channel)
        display = tk.StringVar(value=defaults.display)

        channel_box = ttk.LabelFrame(frame, text="Channel", padding=8)
        channel_box.grid(row=0, column=0, sticky="nsew")
        for value, label in [(ALL_CHANNELS, "All (side by side)")] + [(n, n) for n in CHANNEL_NAMES]:
            ttk.Radiobutton(channel_box, text=label, value=value, variable=channel).pack(anchor="w")

        display_box = ttk.LabelFrame(frame, text="Display", padding=8)
        display_box.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Radiobutton(
            display_box, text="Grayscale intensity", value=DISPLAY_GRAY, variable=display,
        ).pack(anchor="w")
        ttk.Radiobutton(
            display_box, text="Tinted in channel colour", value=DISPLAY_TINTED, variable=display,
        ).pack(anchor="w")

        result: ChannelSplitParams | None = None

        def apply() -> None:
            nonlocal result
            result = ChannelSplitParams(channel=channel.get(), display=display.get())
            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(buttons, text="Apply", command=apply).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)

        window.bind("<Return>", lambda _event: apply())
        window.bind("<Escape>", lambda _event: window.destroy())
        window.grab_set()
        window.wait_window()
        return result
