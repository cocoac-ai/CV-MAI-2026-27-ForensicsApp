from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from forensics_app.core.image_document import ImageDocument
from forensics_app.tools.base import ForensicsTool, ToolResult


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Channel Split"
    category = "Image"
    description = "Split the working image into red, green, and blue channels."
    requires_image = True

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult:
        assert document.current is not None

        img = np.array(document.current)

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        r_img = Image.fromarray(r)
        g_img = Image.fromarray(g)
        b_img = Image.fromarray(b)

        width, height = r_img.size

        output = Image.new("L", (width * 3, height))

        output.paste(r_img, (0, 0))
        output.paste(g_img, (width, 0))
        output.paste(b_img, (width * 2, 0))

        return ToolResult(
            image=output,
            message="Split the image into red, green, and blue channels.",
        )