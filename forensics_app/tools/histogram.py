from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw

from forensics_app.core.image_document import ImageDocument
from forensics_app.tools.base import ForensicsTool, ToolResult

class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Histogram"
    category = "Image"
    description = "Visualize the histogram of the working image."
    requires_image = True

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult:
        assert document.current is not None

        img = np.array(document.current)
        hist, _ = np.histogram(img.ravel(), bins=256)

        width = 256 * 2
        height = 300

        histogram_image = Image.new("RGB", (width, height), "white")

        draw = ImageDraw.Draw(histogram_image)

        max_count = hist.max()

        for i, count in enumerate(hist):
            bar_height = int((count / max_count) * (height - 20))

            x0 = i * 2
            x1 = x0 + 2

            y0 = height - bar_height
            y1 = height

            draw.rectangle(
                (x0, y0, x1, y1),
                fill="black",
            )
        return ToolResult(
            image=histogram_image,
            message="Histogram generated.",
        )