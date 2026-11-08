"""Set 7.1 - Codebook construction.

1. Read the dataset and extract the descriptors of every image, with ORB or SIFT
   (list_dataset, extract_dataset, which uses the core function of Set 5.2).
2. Split the images of every class into train and test (split_dataset).
3. Cluster the train descriptors with K-Means, K = 10 * number of classes (build_codebook).
   Each cluster centre is a visual word; all the centres together are the codebook.
4. Copy the images to the train/<class>/ and test/<class>/ folders inside the dataset
   (save_split).
5. Train an SVM on the histograms of the train images (dataset_histograms, train_classifier),
   and measure it on the test images (score).
6. Save everything for 7.2 and 7.3 (np.savez, pickle).
7. Show the visual words (word_patches, words_figure).
"""

from __future__ import annotations

import json
import pickle
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import numpy as np
from matplotlib.figure import Figure
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from forensics_app.core import ImageDocument

from ..base import ForensicsTool, ToolResult
from ..set5.common import (
    METHOD_SIFT,
    add_extractor_options,
    extractor_details,
    load_image,
    read_extractor_options,
    to_gray,
)
from ..set5.image_lookup import IMAGE_SUFFIXES
from ..set5.keypoints import extract_features
from .common import CLASSIFIER_FILE, DATASET_FOLDER, MODEL_FILE, figure_image, show_figure
from .word_histogram import assign_words, build_histogram

WORDS_PER_CLASS = 10  # K = 10 * num_classes (slide)
TEST_SIZE = 0.25  # 25% of the images of each class are kept for testing
SPLIT_SEED = 42  # change it to get a different train/test split
# The classifier of 7.3 needs a few train images of every class
MIN_IMAGES_PER_CLASS = 8
PATCH_SIZE = 32  # pixels shown for each visual word
PER_ROW = 5  # visual words per row, like the slide
SPLIT_FOLDERS = ("train", "test")  # made by 7.1 inside the dataset folder; they are not classes


# Every image in the dataset: each subfolder is one class, its name is the label
def list_dataset(folder: Path) -> tuple[list[Path], np.ndarray, list[str]]:
    class_names = sorted(
        path.name
        for path in folder.iterdir()
        if path.is_dir() and path.name not in SPLIT_FOLDERS
    )
    paths = []
    labels = []
    for label, name in enumerate(class_names):
        for path in sorted((folder / name).iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path.resolve())
                labels.append(label)
    return paths, np.array(labels), class_names


# Keypoints and descriptors of all the images, skipping the ones that cannot be used.
# Returns the paths and labels that were kept, and one array of each per image.
# Using 5.2 core function
# on_image(done, total) is called after every image, to move the progress bar.
def extract_dataset(
    paths: list[Path],
    labels: np.ndarray,
    method: str,
    settings: dict,
    on_image: Callable[[int, int], None] | None = None,
) -> tuple[list[Path], np.ndarray, list[np.ndarray], list[np.ndarray]]:
    kept = []
    all_keypoints = []
    all_descriptors = []
    for index, path in enumerate(paths):
        if on_image is not None:
            on_image(index + 1, len(paths))
        try:
            keypoints, descriptors = extract_features(
                to_gray(load_image(path)), method, settings
            )
        except (OSError, RuntimeError):
            continue  # not readable, or no keypoints found
        kept.append(index)
        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)
    return [paths[i] for i in kept], labels[kept], all_keypoints, all_descriptors


# Split the images into train and test. Returns the indices of each group, and is_test: one
# True/False per image, which is what the model file keeps for 7.2 and 7.3.
def split_dataset(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # stratify: each class has the same share of images in train and test
    train, test = train_test_split(
        np.arange(len(labels)), test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=labels
    )
    is_test = np.zeros(len(labels), dtype=bool)
    is_test[test] = True
    return train, test, is_test


# Copy every image to folder/train/<class>/ or folder/test/<class>/, so the test images can be
# opened in 7.3. Returns the new path of every image.
def save_split(
    folder: Path,
    paths: list[Path],
    labels: np.ndarray,
    class_names: list[str],
    is_test: np.ndarray,
) -> list[Path]:
    for name in SPLIT_FOLDERS:
        shutil.rmtree(folder / name, ignore_errors=True)  # an older split may have other images
    copies = []
    for path, label, test in zip(paths, labels, is_test):
        part = "test" if test else "train"
        target = folder / part / class_names[label] / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)
        copies.append(target.resolve())
    return copies


# Set7.1: Core function
# Cluster the descriptors (one per row) into k groups; the k centres are the visual words
def build_codebook(descriptors: np.ndarray, k: int) -> np.ndarray:
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0).fit(descriptors)
    return kmeans.cluster_centers_


# Histogram of every image in the dataset, one per row
def dataset_histograms(descriptors: list[np.ndarray], centers: np.ndarray) -> np.ndarray:
    return np.array(
        [build_histogram(assign_words(item, centers), len(centers)) for item in descriptors]
    )


# Set7.1: Core function for training the classifier.
# An SVM alone only says which class wins. The calibration turns its scores into a probability
# per class, so 7.3 can show "p(kangaroo) = 0.64".
def train_classifier(histograms: np.ndarray, labels: np.ndarray) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(SVC(), ensemble=False).fit(histograms, labels)


# A centre is a list of numbers, not a picture. To see a visual word we show the image patch
# around the real descriptor closest to that centre.
def word_patches(
    centers: np.ndarray,
    descriptors: np.ndarray,
    keypoints: np.ndarray,
    owner: np.ndarray,
    paths: list[Path],
) -> list[np.ndarray]:
    half = PATCH_SIZE // 2
    patches = []
    for center in centers:
        closest = np.linalg.norm(descriptors - center, axis=1).argmin()
        gray = to_gray(load_image(paths[owner[closest]]))
        gray = np.pad(gray, half, mode="edge")  # keypoints near the border still get a patch
        # After padding, the pixel (row, col) is at (row + half, col + half), so the patch
        # around it starts at (row, col)
        row, col = keypoints[closest].astype(int)
        patches.append(gray[row : row + PATCH_SIZE, col : col + PATCH_SIZE])
    return patches


# Grid of visual words like the slide
def words_figure(patches: list[np.ndarray]) -> Figure:
    rows = -(-len(patches) // PER_ROW)  # round up
    figure = Figure(figsize=(1.3 * PER_ROW, 1.3 * rows), dpi=100, layout="tight")
    for index, patch in enumerate(patches):
        axes = figure.add_subplot(rows, PER_ROW, index + 1)
        axes.imshow(patch, cmap="gray", vmin=0, vmax=255)
        axes.set_title(f"vw {index}", fontsize=8)
        axes.axis("off")
    return figure


# Small window with a progress bar. SIFT takes about 30 seconds on the Set 7 images;
# without it the app looks frozen.
class ProgressWindow:
    def __init__(self, parent: tk.Misc, title: str) -> None:
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.transient(parent)
        self.window.resizable(False, False)
        frame = ttk.Frame(self.window, padding=16)
        frame.pack(fill="both", expand=True)
        self.label = ttk.Label(frame, text="Starting…", width=45)
        self.label.pack(anchor="w")
        self.bar = ttk.Progressbar(frame, length=320)
        self.bar.pack(pady=(8, 0))
        self.window.grab_set()  # no clicks on the main window while 7.1 is working

    # Show the text and fill the bar to done / total
    def show(self, text: str, done: int, total: int) -> None:
        self.label["text"] = text
        self.bar["maximum"] = total
        self.bar["value"] = done
        self.window.update()  # draw it now; the computation does not give Tk time to draw

    def close(self) -> None:
        if self.window.winfo_exists():
            self.window.destroy()


class CodebookTool(ForensicsTool):
    tool_id = "bow_codebook"
    title = "Codebook (visual words)"
    category = "Set 7 · BOW"
    description = "Cluster the descriptors of a dataset with K-Means and show the visual words."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        params = self._ask_params(parent)
        if params is None:
            return None  # the user pressed Cancel
        folder, method, settings = params

        progress = ProgressWindow(parent, self.title)
        try:
            return self._build(parent, folder, method, settings, progress)
        finally:
            progress.close()  # also when something fails

    def _build(
        self, parent: tk.Misc, folder: Path, method: str, settings: dict, progress: ProgressWindow
    ) -> ToolResult:
        # Step 1: read the dataset and extract the descriptors (the slow step)
        paths, labels, class_names = list_dataset(folder)

        def on_image(done: int, total: int) -> None:
            progress.show(f"Step 1/5: {method} descriptors, image {done} / {total}", done, total)

        paths, labels, keypoints, descriptors = extract_dataset(
            paths, labels, method, settings, on_image
        )

        counts = np.bincount(labels, minlength=len(class_names))
        if len(class_names) < 2 or counts.min() < MIN_IMAGES_PER_CLASS:
            return ToolResult(
                message=(
                    "The folder needs at least 2 subfolders (one per class) with "
                    f"{MIN_IMAGES_PER_CLASS} usable images each."
                )
            )

        # Step 2: split into train and test
        train, test, is_test = split_dataset(labels)

        # Step 3: build the codebook.
        # Same as the slide: stack the descriptors of all the images in one array.
        # Only the train images build the codebook, so the test images stay unseen.
        train_descriptors = np.vstack([descriptors[i] for i in train]).astype(float)
        train_keypoints = np.vstack([keypoints[i] for i in train])
        train_owner = np.repeat(train, [len(descriptors[i]) for i in train])

        k = WORDS_PER_CLASS * len(class_names)
        if len(train_descriptors) < k:
            return ToolResult(message=f"Only {len(train_descriptors)} descriptors for K = {k}.")
        progress.show(f"Step 2/5: K-Means with K = {k}", 1, 5)
        centers = build_codebook(train_descriptors, k)

        # Step 4: copy the images to the train and test folders
        progress.show("Step 3/5: copying the train and test images", 2, 5)
        paths = save_split(folder, paths, labels, class_names, is_test)

        # Step 5: train the SVM on the train images, measure it on the test images
        progress.show("Step 4/5: training the SVM", 3, 5)
        histograms = dataset_histograms(descriptors, centers)
        classifier = train_classifier(histograms[~is_test], labels[~is_test])
        accuracy = classifier.score(histograms[is_test], labels[is_test])

        # Step 6: save the model.
        # Saved as they are (uint8 or bool): as floats the file would be 8 times bigger
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            MODEL_FILE,
            method=method,
            settings=json.dumps(settings),
            class_names=np.array(class_names),
            paths=np.array([str(path) for path in paths]),
            labels=labels,
            is_test=is_test,
            accuracy=accuracy,
            centers=centers,
            keypoints=np.vstack(keypoints),
            descriptors=np.vstack(descriptors),
            owner=np.repeat(np.arange(len(paths)), [len(item) for item in descriptors]),
        )
        CLASSIFIER_FILE.write_bytes(pickle.dumps(classifier))

        # Step 7: show the visual words
        progress.show("Step 5/5: drawing the visual words", 4, 5)
        patches = word_patches(centers, train_descriptors, train_keypoints, train_owner, paths)
        figure = words_figure(patches)
        progress.close()  # before the figure window opens
        show_figure(parent, figure, f"Visual words (K = {k})")

        details = {"Operation": "Codebook construction", "Folder": str(folder)}
        details.update(extractor_details(method, settings))
        details["Classes"] = ", ".join(class_names)
        details["Train images"] = len(train)
        details["Test images"] = len(test)
        details["Train descriptors"] = str(train_descriptors.shape)
        details["Visual words K"] = k
        details["Classifier"] = "SVM (RBF kernel)"
        details["Accuracy on the test images"] = f"{accuracy:.1%}"
        details["Train/test folders"] = str(folder)
        details["Model file"] = str(MODEL_FILE)
        return ToolResult(
            message=(
                f"Codebook of {k} words and SVM built from {len(train)} images; model saved. "
                f"For 7.3, open an image from {folder / 'test'}."
            ),
            details=details,
            preview=figure_image(figure),
        )

    def _ask_params(self, parent: tk.Misc) -> tuple[Path, str, dict] | None:
        window = tk.Toplevel(parent)
        window.title(self.title)
        window.transient(parent)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)

        folder_box = ttk.LabelFrame(
            frame, text="Dataset folder (one subfolder per class)", padding=8
        )
        folder_box.pack(fill="both")
        folder_entry = ttk.Entry(folder_box, width=40)
        folder_entry.pack(side="left")
        folder_entry.insert(0, str(DATASET_FOLDER))

        def browse() -> None:
            path = filedialog.askdirectory(parent=window, title="Choose the dataset folder")
            if path:
                folder_entry.delete(0, "end")
                folder_entry.insert(0, path)

        ttk.Button(folder_box, text="Browse…", command=browse).pack(side="left", padx=(6, 0))

        method, entries = add_extractor_options(frame)
        method.set(METHOD_SIFT)  # clearly better than ORB on the Set 7 images

        ttk.Label(frame, text="SIFT can take about 30 seconds on the Set 7 images.").pack(
            anchor="w", pady=(8, 0)
        )

        result = None

        def apply() -> None:
            nonlocal result
            folder = Path(folder_entry.get().strip())
            if not folder.is_dir():
                messagebox.showerror("Invalid path", "Please choose a folder.", parent=window)
                return
            settings = read_extractor_options(entries)
            if settings is None:
                messagebox.showerror(
                    "Invalid value",
                    "Please enter numbers above 0 (whole numbers, except sigma min).",
                    parent=window,
                )
                return
            result = (folder, method.get(), settings)
            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.pack(pady=(12, 0))
        ttk.Button(buttons, text="Apply", command=apply).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)

        window.bind("<Return>", lambda _event: apply())
        window.bind("<Escape>", lambda _event: window.destroy())
        window.grab_set()
        window.wait_window()
        return result
