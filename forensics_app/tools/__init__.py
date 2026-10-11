"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .convolution import ConvolutionTool
from .image_filters import ImageFiltersTool
from .canny_edge import CannyTool
from .sharpening import SharpeningTool
from .contrast_stretching import ContrastStretchingTool

def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ConvolutionTool(),
            ImageFiltersTool(),
            CannyTool(),
            SharpeningTool(),
            ContrastStretchingTool(),
            HistogramTool(),
            ChannelSplitTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]