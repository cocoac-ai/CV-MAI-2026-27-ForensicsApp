from __future__ import annotations

import tkinter as tk

from PIL import ImageOps

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

class ChannelSplit(ForensicsTool):
    tool_id = "channel split"
    title = "Split the channel"
    category = "Starter tools"
    description = "Split the working image to channels."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window
        output = ImageOps.split(document.current)
        return ToolResult(
            image=output,
            message="Converted the image to grayscale.",
            details={"Operation": "Grayscale", "Output mode": output.mode},
        )