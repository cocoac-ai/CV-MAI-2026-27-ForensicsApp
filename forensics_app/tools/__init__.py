"""Register course functionality here so it appears in the sidebar."""

from .registry import ToolRegistry

from .set1.image_info import ImageInfoTool
from .set1.grayscale import GrayscaleTool

from .set2.channel_split import ChannelSplitTool
from .set2.channel_swap import ChannelSwapTool
from .set2.masking import MaskingTool
from .set2.histogram import HistogramTool
from .set2.contrast_stretching import ContrastStretchingTool

from .set3.convolution import ConvolutionTool
from .set3.image_filters import ImageFiltersTool
from .set3.canny_edge import CannyTool
from .set3.edge_visualization import EdgeVisualizationTool
from .set3.sharpening import SharpeningTool

from .set4.crosscorrelation import NormalizedCrossCorrelationTool
from .set4.template_matching_configs import TemplateMatchingConfigsTool
from .set4.hog import HogTool
from .set4.person_detection import PersonDetectionTool


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

            NormalizedCrossCorrelationTool(),
            TemplateMatchingConfigsTool(),
            HogTool(),
            PersonDetectionTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]