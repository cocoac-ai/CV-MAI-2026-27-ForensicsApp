"""Register course functionality here so it appears in the sidebar."""

from .registry import ToolRegistry

from .image_info import ImageInfoTool
from .grayscale import GrayscaleTool

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .masking import MaskingTool
from .histogram import HistogramTool
from .contrast_stretching import ContrastStretchingTool

from .convolution import ConvolutionTool
from .image_filters import ImageFiltersTool
from .canny_edge import CannyTool
from .edge_visualization import EdgeVisualizationTool
from .sharpening import SharpeningTool

from .hog import HogTool
from .person_detection import PersonDetectionTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),

            ChannelSplitTool(),
            ChannelSwapTool(),
            MaskingTool(),
            HistogramTool(),
            ContrastStretchingTool(),

            ConvolutionTool(),
            ImageFiltersTool(),
            CannyTool(),
            EdgeVisualizationTool(),
            SharpeningTool(),

            HogTool(),
            PersonDetectionTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]