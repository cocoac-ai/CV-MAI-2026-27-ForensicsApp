from __future__ import annotations

import numpy as np
from PIL import Image

from forensics_app.core.image_document import ImageDocument
from forensics_app.tools.base import ForensicsTool, ToolResult


class MaskingTool(ForensicsTool):
    tool_id = "masking"
    title = "Masking"
    category = "Image"
    description = "Apply a threshold mask to the working image."
    requires_image = True

    def run(
        self,
        parent,
        document: ImageDocument,
    ) -> ToolResult:
        assert document.current is not None

        image = np.array(document.current)

        mask = image > 135

        masked = image * mask

        output = Image.fromarray(masked)

        return ToolResult(
            image=output,
            message="Applied mask with threshold 135.",
        )