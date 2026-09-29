"""Register course functionality here so it appears in the sidebar."""

from .grayscale import GrayscaleTool
from .channel_split import ChannelSplit

from .image_info import ImageInfoTool
from .registry import ToolRegistry


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ChannelSplit(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
