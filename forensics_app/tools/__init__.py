"""Register course functionality here so it appears in the sidebar."""

from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .convolution import ConvolutionTool
from .image_filters import ImageFiltersTool
from .canny_edge import CannyTool

def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ConvolutionTool(),
            ImageFiltersTool(),
            CannyTool()
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
