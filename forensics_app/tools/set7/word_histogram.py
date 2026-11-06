"""Set 7.2 - Histogram construction.

1. Load the model of 7.1 and choose one random test image of each class (load_model).
2. Assign features to words: map each descriptor to its closest cluster centre (assign_words).
3. Count how many descriptors fell on each word: a K-dimensional frequency histogram
   (build_histogram).
4. Show the histograms, like the slide (draw_image_and_histogram, show_figure).

7.3 imports the core functions from here.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from forensics_app.core import ImageDocument

from ..base import ForensicsTool, ToolResult
from ..set5.common import load_image
from .common import (
    NO_MODEL_MESSAGE,
    draw_image_and_histogram,
    figure_image,
    image_features,
    load_model,
    show_figure,
)


# Set7.2: Core function for assigning.
# For each descriptor, the index of its closest centre (Euclidean distance)
def assign_words(descriptors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # distances[i, j] = distance between descriptor i and centre j
    distances = np.linalg.norm(descriptors[:, None, :] - centers[None, :, :], axis=2)
    return distances.argmin(axis=1)


# Set7.2: Core function for the histogram.
# Normalized, so an image with many keypoints can be compared with one with few
def build_histogram(words: np.ndarray, k: int) -> np.ndarray:
    counts = np.bincount(words, minlength=k)
    return counts / counts.sum()


class WordHistogramTool(ForensicsTool):
    tool_id = "bow_histogram"
    title = "Histogram of visual words"
    category = "Set 7 · BOW"
    description = "Show the visual word histogram of one random test image of each class."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        # Step 1: load the model; the loop below chooses the images
        model = load_model()
        if model is None:
            return ToolResult(message=NO_MODEL_MESSAGE)

        centers = model["centers"]
        class_names = model["class_names"]
        rng = np.random.default_rng()  # a new choice of images every time

        figure = Figure(figsize=(9, 2.6 * len(class_names)), dpi=100, layout="tight")
        details = {"Operation": "Histogram construction", "Visual words K": len(centers)}
        for row, name in enumerate(class_names):
            # Only test images, the ones the codebook has never seen
            candidates = np.flatnonzero(model["is_test"] & (model["labels"] == row))
            index = rng.choice(candidates)
            path = Path(str(model["paths"][index]))

            # Step 2: assign the descriptors to words
            keypoints, descriptors = image_features(model, index)
            words = assign_words(descriptors, centers)
            # Step 3: count the words
            histogram = build_histogram(words, len(centers))
            # Step 4: draw this image and its histogram
            draw_image_and_histogram(
                figure, row, len(class_names), load_image(path), keypoints, words, histogram, name
            )
            details[f"{name}: image"] = path.name
            details[f"{name}: most frequent word"] = f"vw {histogram.argmax()}"

        show_figure(parent, figure, self.title)
        return ToolResult(
            message="Histograms of test images opened in a new window.",
            details=details,
            preview=figure_image(figure),
        )
