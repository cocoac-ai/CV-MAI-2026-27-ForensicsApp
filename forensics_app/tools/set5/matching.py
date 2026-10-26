"""Set 5.3 - Match ORB or SIFT descriptors with a second image.

5.4 imports the core function match_features from here.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import numpy as np
from skimage import feature

from forensics_app.core import ImageDocument

from ..base import ForensicsTool, ToolResult
from .common import (
    METHOD_ORB,
    ask_match_params,
    draw_matches,
    load_image,
    match_details,
    shrink,
    to_gray,
)
from .keypoints import extract_features


# Set5.3: Core function for matching.
# Each row of the result is (index in desc1, index in desc2) of one match
def match_features(desc1: np.ndarray, desc2: np.ndarray, method: str, settings: dict) -> np.ndarray:
    # ORB descriptors are bits, so they are compared with the Hamming distance
    metric = "hamming" if method == METHOD_ORB else "euclidean"
    return feature.match_descriptors(
        desc1,
        desc2,
        cross_check=settings["cross_check"],
        max_ratio=settings["max_ratio"],
        metric=metric,
    )


class MatchingTool(ForensicsTool):
    tool_id = "feature_matching"
    title = "Match with second image"
    category = "Set 5 · SIFT"
    description = "Match ORB or SIFT keypoints of the image with a second image."
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window

        params = ask_match_params(parent, "Match with second image", pick_folder=False)
        if params is None:
            return None  # the user pressed Cancel
        second_path, method, settings = params

        image1 = shrink(document.current)
        image2 = load_image(second_path)
        try:
            keypoints1, desc1 = extract_features(to_gray(image1), method, settings)
            keypoints2, desc2 = extract_features(to_gray(image2), method, settings)
        except RuntimeError as exc:
            # skimage raises this when it finds no keypoints at all
            return ToolResult(message=f"{method} failed: {exc}")

        matches = match_features(desc1, desc2, method, settings)
        output = draw_matches(image1, keypoints1, image2, keypoints2, matches)

        details = {"Operation": "Feature matching", "Second image": Path(second_path).name}
        details.update(match_details(method, settings))
        details["Keypoints (image 1)"] = len(keypoints1)
        details["Keypoints (image 2)"] = len(keypoints2)
        details["Matches"] = len(matches)

        return ToolResult(
            image=output,
            message=f"{method}: {len(matches)} matches with {Path(second_path).name}.",
            details=details,
        )
