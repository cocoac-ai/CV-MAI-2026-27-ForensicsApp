
from .registry import ToolRegistry

# Set 1
from .set1.image_info import ImageInfoTool
from .set1.grayscale import GrayscaleTool

# Set 2
from .set2.channel_split import ChannelSplitTool
from .set2.channel_swap import ChannelSwapTool
from .set2.masking import MaskingTool
from .set2.histogram import HistogramTool
from .set2.contrast_stretching import ContrastStretchingTool

# Set 3
from .set3.convolution import ConvolutionTool
from .set3.image_filters import ImageFiltersTool
from .set3.canny import CannyTool
from .set3.edge_visualization import EdgeVisualizationTool
from .set3.sharpening import SharpeningTool

# Set 4
from .set4.normalized_cross_correlation import NormalizedCrossCorrelationTool
from .set4.template_matching_configs import TemplateMatchingConfigsTool
from .set4.hog import HogTool
from .set4.person_detection import PersonDetectionTool

# Set 5
from .set5.corner_detection import CornerDetectionTool
from .set5.keypoints import KeypointsTool
from .set5.matching import MatchingTool
from .set5.image_lookup import ImageLookupTool

# Set 6
from .set6.face_detection import FaceDetectionTool
from .set6.eigenface import EigenfacesTool
from .set6.face_recognition import FaceRecognitionTool

# Set 7
from .set7.codebook import CodebookTool
from .set7.word_histogram import WordHistogramTool
from .set7.classification import ClassificationTool

# Set 8
from .set8.classification import CnnClassificationTool
from .set8.projection import FeatureProjectionTool

# Set 9
from .set9.yolo_detection import (
    PretrainedYOLOTool,
    BloodCellDetectionTool,
)


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            # Set 1
            ImageInfoTool(),
            GrayscaleTool(),

            # Set 2
            ChannelSplitTool(),
            ChannelSwapTool(),
            MaskingTool(),
            HistogramTool(),
            ContrastStretchingTool(),

            # Set 3
            ConvolutionTool(),
            ImageFiltersTool(),
            CannyTool(),
            EdgeVisualizationTool(),
            SharpeningTool(),

            # Set 4
            NormalizedCrossCorrelationTool(),
            TemplateMatchingConfigsTool(),
            HogTool(),
            PersonDetectionTool(),

            # Set 5
            CornerDetectionTool(),
            KeypointsTool(),
            MatchingTool(),
            ImageLookupTool(),

            # Set 6
            FaceDetectionTool(),
            EigenfacesTool(),
            FaceRecognitionTool(),

            # Set 7
            CodebookTool(),
            WordHistogramTool(),
            ClassificationTool(),

            # Set 8
            CnnClassificationTool(),
            FeatureProjectionTool(),

            # Set 9
            PretrainedYOLOTool(),
            BloodCellDetectionTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
