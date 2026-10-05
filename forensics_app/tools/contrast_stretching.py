from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image
from skimage import exposure

from forensics_app.core.image_document import ImageDocument
from forensics_app.tools.base import ForensicsTool, ToolResult

class ContrastStretchingTool(ForensicsTool):
    tool_id = "contrast_stretching"
    title = "Contrast Stretching"
    category = "Image"
    description = "Improve image contrast by stretching the intensity range."
    requires_image = True

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult:
        
        assert document.current is not None

        img = np.array(document.current)
        p2, p98 = np.percentile(img, (2, 98))

        stretched = exposure.rescale_intensity(
            img,
            in_range = (p2, p98),
            out_range = (0, 1),
        )
        stretched = (stretched * 255).astype(np.uint8)
        stretched_image = Image.fromarray(stretched)

        return ToolResult(
            image = stretched_image,
            message = "Contrast stretching applied.",
        )