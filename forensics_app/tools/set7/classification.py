"""Set 7.3 - Classification of a new image.

1. Load the model of 7.1, with the trained SVM (load_model).
2. Histogram of the new image (extract_features, assign_words, build_histogram).
3. Show the histogram of the new image (_open_window).
4. After clicking "Classify": run the classifier and show the top probability (classify).
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV

from forensics_app.core import ImageDocument

from ..base import ForensicsTool, ToolResult
from ..set5.common import shrink, to_gray
from ..set5.keypoints import extract_features
from .common import (
    NO_MODEL_MESSAGE,
    draw_image_and_histogram,
    load_model,
    model_method,
    show_figure,
)
from .word_histogram import assign_words, build_histogram


# "class/file.jpg": the same for an image in the dataset and for its copy in train/ or test/
def class_and_file(path: str | Path) -> str:
    return "/".join(Path(path).parts[-2:])


# Set7.3: Core function for classifying.
# Returns the most likely class and its probability
def classify(classifier: CalibratedClassifierCV, histogram: np.ndarray) -> tuple[int, float]:
    probabilities = classifier.predict_proba(histogram[None, :])[0]  # the SVM wants one row
    best = probabilities.argmax()
    return int(classifier.classes_[best]), float(probabilities[best])


class ClassificationTool(ForensicsTool):
    tool_id = "bow_classification"
    title = "Classify image"
    category = "Set 7 · BOW"
    description = (
        "Open an image from set7-images/test first (the SVM has not seen it). "
        "Show its visual word histogram, then classify it."
    )
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window

        # Step 1: load the model
        model = load_model()
        if model is None:
            return ToolResult(message=NO_MODEL_MESSAGE)
        method, settings = model_method(model)

        # Step 2: the new image goes through the same steps as the dataset images
        image = shrink(document.current)
        try:
            keypoints, descriptors = extract_features(to_gray(image), method, settings)
        except RuntimeError as exc:
            # skimage raises this when it finds no keypoints at all
            return ToolResult(message=f"{method} failed: {exc}")
        centers = model["centers"]
        words = assign_words(descriptors, centers)
        histogram = build_histogram(words, len(centers))

        # Step 3: show the histogram
        self._open_window(
            parent, image, keypoints, words, histogram, model["classifier"], model["class_names"]
        )

        # A train image would look easier than it is: the SVM has already seen it
        name = document.path.name if document.path else "working image"
        path = class_and_file(document.path) if document.path else ""
        is_test = model["is_test"]
        train_images = [class_and_file(item) for item in model["paths"][~is_test]]
        if path in train_images:
            messagebox.showwarning(
                self.title,
                "The SVM was trained on this image, so the result looks better than it is.\n"
                "Open an image from set7-images/test instead.",
                parent=parent,
            )
        return ToolResult(
            message="Histogram opened in a new window; click Classify to see the result.",
            details={
                "Operation": "Classification",
                "Image": name,
                "Used for training": path in train_images,
                "Method": method,
                "Keypoints found": len(keypoints),
                "Classifier": "SVM (RBF kernel)",
                "Train images": int((~is_test).sum()),
                "Test images": int(is_test.sum()),
                "Accuracy on the test images": f"{float(model['accuracy']):.1%}",
            },
        )

    def _open_window(
        self,
        parent: tk.Misc,
        image: Image.Image,
        keypoints: np.ndarray,
        words: np.ndarray,
        histogram: np.ndarray,
        classifier: CalibratedClassifierCV,
        class_names: np.ndarray,
    ) -> None:
        figure = Figure(figsize=(9, 3.6), dpi=100, layout="tight")
        image_axes = draw_image_and_histogram(
            figure, 0, 1, image, keypoints, words, histogram, "New image"
        )

        # Step 4: runs when the user clicks "Classify"
        def on_classify() -> None:
            label, probability = classify(classifier, histogram)
            image_axes.set_title(f"p({class_names[label]}) = {probability:.2f}")
            figure.canvas.draw_idle()

        show_figure(parent, figure, self.title, on_classify)
