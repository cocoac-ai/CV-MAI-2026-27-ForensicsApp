"""Set 8.2 - Load a pretrained CNN and classify an image.

The models are trained in the Set 8 notebooks (Set 8.1, in Colab or locally) and saved in
models/set8/colab/ or models/set8/local/: a .onnx file (the network) and a .json file
(its passport).
1. Choose a model: the small CNN or InceptionV3 (transfer learning), trained in Colab or locally.
2. Load its passport and the network. onnxruntime runs the .onnx file, so the app does
   not need tensorflow.
3. Prepare the image exactly as the passport says (RGB, size, value range).
4. Run the network: one probability per class.
5. Show the image, the probabilities and the training history; the passport goes in the details.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import onnxruntime as ort
from matplotlib.figure import Figure
from PIL import Image

from forensics_app.core import ImageDocument

from ..base import ForensicsTool, ToolResult
from ..set7.common import figure_image
from .common import TRAINED_IN

# Name shown in the dialog -> file name (without .onnx / .json)
NETWORKS = {
    "Shallow CNN": "set8_shallow_cnn",  # 1 conv layer
    "Small CNN": "set8_small_cnn",
    "VGG": "set8_vgg",  # deeper, input 64x64
    "InceptionV3": "set8_inception",  # transfer learning
}
NO_MODEL_MESSAGE = "No Set 8 model found in models/set8/colab or models/set8/local."


# Every model that has both files, for example
# "Small CNN (Colab)" -> models/set8/colab/set8_small_cnn (+ .onnx / .json)
def available_models() -> dict[str, Path]:
    models = {}
    for network, file in NETWORKS.items():
        for where, folder in TRAINED_IN.items():
            base = folder / file
            if base.with_suffix(".onnx").is_file() and base.with_suffix(".json").is_file():
                models[f"{network} ({where})"] = base
    return models


# Set8.2: Core function for preprocessing.
# The image must look like the training images, so follow the passport.
# Returns a batch with one image: shape (1, height, width, 3).
def prepare_image(image: Image.Image, preprocessing: dict) -> np.ndarray:
    rgb = image.convert(preprocessing["color_order"])
    # The models learned from 32x32 CIFAR-10 images: make the image that size first,
    # then as big as the model wants (InceptionV3 was trained on 32x32 images made bigger)
    height, width = preprocessing["original_size"]
    rgb = rgb.resize((width, height), Image.BILINEAR)
    height, width = preprocessing["input_size"]
    rgb = rgb.resize((width, height), Image.BILINEAR)

    pixels = np.asarray(rgb, dtype=np.float32) / 255  # [0, 255] -> [0, 1]
    low, high = preprocessing["value_range"]
    pixels = pixels * (high - low) + low  # [0, 1] -> [low, high]
    return pixels[None]  # the network wants a batch: add the batch axis


# Set8.2: Core function for classifying.
# The last layer is a softmax, so the output is already one probability per class.
def classify(session: ort.InferenceSession, batch: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: batch})[0][0]


# Training history from the passport as one curve per value (loss, accuracy, ...).
# InceptionV3 was trained in two steps; their epochs are put one after the other.
def history_curves(history: dict) -> dict[str, list[float]]:
    if "accuracy" in history:
        return history
    curves: dict[str, list[float]] = {}
    for step in history.values():
        for name, values in step.items():
            curves.setdefault(name, []).extend(values)
    return curves


class CnnClassificationTool(ForensicsTool):
    tool_id = "cnn_classification"
    title = "Classify image (CNN)"
    category = "Set 8 · CNN"
    description = (
        "Load a CNN trained in the Set 8 notebook, show its passport and classify the "
        "current image into one of the 10 CIFAR-10 classes. Try images/set8-images."
    )
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window

        # Step 1: choose the model
        models = available_models()
        if not models:
            return ToolResult(message=NO_MODEL_MESSAGE)
        name = self._ask_model(parent, list(models))
        if name is None:
            return None  # the user pressed Cancel

        # Step 2: load the passport and the network
        base = models[name]
        passport = json.loads(base.with_suffix(".json").read_text())
        session = ort.InferenceSession(str(base.with_suffix(".onnx")))

        # Steps 3 and 4: prepare the image and classify it
        preprocessing = passport["preprocessing"]
        batch = prepare_image(document.current, preprocessing)
        probabilities = classify(session, batch)
        class_names = passport["class_names"]
        best = int(probabilities.argmax())

        # Step 5: show the result and the passport
        figure = self._draw(document.current, probabilities, class_names, passport)
        details = {
            "Operation": "CNN classification",
            "Image": document.path.name if document.path else "working image",
            "Predicted": f"{class_names[best]} ({probabilities[best]:.1%})",
        }
        # The test images are saved as <class>_<number>.png
        if document.path is not None:
            true_name = document.path.stem.rsplit("_", 1)[0]
            if true_name in class_names:
                details["True"] = true_name
        details.update(
            {
                "Model": passport["model"],
                "Model file": f"{base.parent.name}/{base.name}.onnx",
                "Architecture": passport["architecture"],
                "Dataset": passport["dataset"],
                "Train samples": passport["train_samples"],
                "Validation samples": passport["validation_samples"],
                "Test samples": passport["test_samples"],
                "Classes": ", ".join(class_names),
                "Color order": preprocessing["color_order"],
                "Input size": "x".join(map(str, preprocessing["input_size"])),
                "Pixel values": preprocessing["scale"],
            }
        )
        for key, value in passport["training"].items():
            details[f"Training: {key}"] = value
        details["Test accuracy"] = f"{passport['test_accuracy']:.1%}"

        return ToolResult(
            message=f"{name}: {class_names[best]} ({probabilities[best]:.1%}).",
            details=details,
            preview=figure_image(figure),
        )

    def _draw(
        self,
        image: Image.Image,
        probabilities: np.ndarray,
        class_names: list[str],
        passport: dict,
    ) -> Figure:
        figure = Figure(figsize=(12, 3.6), dpi=100, layout="tight")
        best = int(probabilities.argmax())

        image_axes = figure.add_subplot(1, 3, 1)
        # "nearest" keeps the pixels of a 32x32 image sharp instead of blurring them
        image_axes.imshow(image.convert("RGB"), interpolation="nearest")
        image_axes.set_title(f"Predicted: {class_names[best]} ({probabilities[best]:.1%})")
        image_axes.axis("off")

        bars_axes = figure.add_subplot(1, 3, 2)
        colors = ["tab:orange" if i == best else "lightgray" for i in range(len(class_names))]
        bars_axes.barh(class_names, probabilities, color=colors, edgecolor="black", linewidth=0.5)
        bars_axes.invert_yaxis()  # first class at the top
        bars_axes.set_xlim(0, 1)
        bars_axes.set_title("Probability of each class")

        history_axes = figure.add_subplot(1, 3, 3)
        curves = history_curves(passport["history"])
        epochs = np.arange(1, len(curves["accuracy"]) + 1)
        history_axes.plot(epochs, curves["accuracy"], label="train")
        history_axes.plot(epochs, curves["val_accuracy"], label="validation")
        if "step_1" in passport["history"]:  # InceptionV3: mark where fine-tuning starts
            start = len(passport["history"]["step_1"]["accuracy"]) + 0.5
            history_axes.axvline(start, color="gray", linestyle="--", label="fine-tuning starts")
        history_axes.set_title(f"Training history ({passport['model']})")
        history_axes.set_xlabel("Epoch")
        history_axes.set_ylabel("Accuracy")
        history_axes.legend()
        return figure

    def _ask_model(self, parent: tk.Misc, names: list[str]) -> str | None:
        """Small dialog asking which model to use."""
        window = tk.Toplevel(parent)
        window.title(self.title)
        window.transient(parent)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Model:").grid(row=0, column=0, sticky="w")
        choice = ttk.Combobox(frame, values=names, state="readonly", width=36)
        choice.grid(row=0, column=1, padx=6)
        choice.set(names[0])

        result = None

        def apply() -> None:
            nonlocal result
            result = choice.get()
            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(buttons, text="Classify", command=apply).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)

        window.bind("<Return>", lambda _event: apply())
        window.bind("<Escape>", lambda _event: window.destroy())
        window.grab_set()
        window.wait_window()
        return result
