"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .contrast_stretching import ContrastStretchingTool

def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ContrastStretchingTool(),
            HistogramTool(),
            ChannelSplitTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]