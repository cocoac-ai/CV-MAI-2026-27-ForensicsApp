"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .convolution import ConvolutionTool
from .image_filters import ImageFiltersTool
from .canny_edge import CannyTool
<<<<<<< HEAD
from .sharpening import SharpeningTool
from .contrast_stretching import ContrastStretchingTool
=======
from .crosscorrelation import NormalizedCrossCorrelationTool
from .template_matching_configs import TemplateMatchingConfigsTool
>>>>>>> 6e0a156 (Add Set 4.1 and 4.2 template matching tools)

def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ConvolutionTool(),
            ImageFiltersTool(),
            CannyTool(),
<<<<<<< HEAD
            SharpeningTool(),
            ContrastStretchingTool(),
            HistogramTool(),
            ChannelSplitTool(),
=======
            NormalizedCrossCorrelationTool(),
            TemplateMatchingConfigsTool()
>>>>>>> 6e0a156 (Add Set 4.1 and 4.2 template matching tools)
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]