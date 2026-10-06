from __future__ import annotations
import tkinter as tk
import numpy as np

from PIL import Image

from forensics_app.core.image_document import ImageDocument

from forensics_app.tools.base import ForensicsTool, ToolResult

class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Channel Swap"
    category = "Image"
    description = "Swaps the Image Color Channels"
    requires_image = True

    def run(
        self, 
        parent: tk.Misc,
        document: ImageDocument,
    )-> ToolResult:
        assert document.current is not None
        img = np.array(document.current)
        tmp = img[:, :, 0].copy()
        img[:, :, 0] = img[:, :, 1]
        img[:, :, 1] = tmp
        output = Image.fromarray(img)
        return ToolResult(
            image=output,
            message="Swapped the red and green channels."
    )