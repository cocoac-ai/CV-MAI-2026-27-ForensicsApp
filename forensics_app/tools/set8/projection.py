"""Set 8.3 - Feature projection with t-SNE and UMAP.

The Set 8 notebooks give every test image to a model of 8.1 and keep the values of the layer
before its classifier: one feature vector per image (small CNN: its 'features' layer, 128 values;
InceptionV3: its Dense(1024) layer). They are saved in models/set8-colab/ or models/set8-local/,
one .npz file per model.
1. Choose the features (which model, trained in Colab or locally); load them and the true class
   of each image.
2. Keep N images per class, so the projection takes seconds instead of minutes.
3. Project the 128-d vectors onto 2D with t-SNE and with UMAP.
4. Plot both, one color per class: if the CNN learned good features, each class is a group.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.figure import Figure
from sklearn.manifold import TSNE

from forensics_app.core import ImageDocument

from ..base import ForensicsTool, ToolResult
from ..set7.common import figure_image, show_figure
from .common import TRAINED_IN

# Model shown in the dialog -> its features file (small CNN first: it is the default)
FEATURE_FILES = {
    "Small CNN": "set8_test_features.npz",  # 'features' layer, 128 values
    "InceptionV3": "set8_test_features_inception.npz",  # 1024 values
}
NO_FEATURES_MESSAGE = "No features found (set8_test_features*.npz in models/set8-colab or local)."
DEFAULT_PER_CLASS = 200  # 10 classes x 200 = 2000 points: a few seconds
MAX_PER_CLASS = 1000  # the CIFAR-10 test set has 1000 images per class


# Every features file that exists, for example
# "InceptionV3 (Colab)" -> models/set8-colab/set8_test_features_inception.npz
def available_features() -> dict[str, Path]:
    files = {}
    for network, file in FEATURE_FILES.items():
        for where, folder in TRAINED_IN.items():
            if (folder / file).is_file():
                files[f"{network} ({where})"] = folder / file
    return files


# The first n images of each class (the same ones every time)
def pick_per_class(labels: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([np.flatnonzero(labels == c)[:n] for c in np.unique(labels)])


# Set8.3: Core function for t-SNE.
# Points that are close in 128-d stay close in 2D; distances between groups mean little.
def project_tsne(features: np.ndarray) -> np.ndarray:
    return TSNE(n_components=2, random_state=0).fit_transform(features)


# Set8.3: Core function for UMAP.
# Like t-SNE, but faster and it keeps more of the large-scale layout.
def project_umap(features: np.ndarray) -> np.ndarray:
    import umap  # imported here: it is slow to import, and only this tool needs it

    # n_jobs=1 and compatibility_layout=True are what UMAP already does here;
    # writing them out only silences its two warnings.
    reducer = umap.UMAP(n_components=2, random_state=0, n_jobs=1, compatibility_layout=True)
    return reducer.fit_transform(features)


class FeatureProjectionTool(ForensicsTool):
    tool_id = "feature_projection"
    title = "Feature projection (t-SNE, UMAP)"
    category = "Set 8 · CNN"
    description = (
        "Project the CNN feature vectors of the CIFAR-10 test images onto 2D "
        "with t-SNE and UMAP, one color per class."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        # Step 1: choose and load the feature vectors
        files = available_features()
        if not files:
            return ToolResult(message=NO_FEATURES_MESSAGE)
        params = self._ask_params(parent, list(files))
        if params is None:
            return None  # the user pressed Cancel
        source, n = params
        with np.load(files[source]) as data:
            features, labels, class_names = data["features"], data["labels"], data["class_names"]

        # Step 2: keep n images per class
        chosen = pick_per_class(labels, n)
        features, labels = features[chosen], labels[chosen]

        # Step 3: project onto 2D (this takes a few seconds: show a busy cursor)
        parent.config(cursor="watch")
        parent.update()
        try:
            projections = {"t-SNE": project_tsne(features), "UMAP": project_umap(features)}
        finally:
            parent.config(cursor="")

        # Step 4: plot both projections
        figure = Figure(figsize=(12, 5.5), dpi=100, layout="tight")
        figure.suptitle(f"Features of {source}")
        for index, (name, points) in enumerate(projections.items(), start=1):
            axes = figure.add_subplot(1, 2, index)
            for c, class_name in enumerate(class_names):
                rows = labels == c
                axes.scatter(points[rows, 0], points[rows, 1], s=4, label=class_name)
            axes.set_title(name)
            axes.set_xlabel(f"{name} 1")
            axes.set_ylabel(f"{name} 2")
            axes.set_xticks([])  # the axes of a projection have no unit
            axes.set_yticks([])
        axes.legend(markerscale=3, fontsize=8, loc="best")

        show_figure(parent, figure, self.title)
        return ToolResult(
            message="t-SNE and UMAP projections opened in a new window.",
            details={
                "Operation": "Feature projection",
                "Features of": source,
                "Features": f"{features.shape[1]} values per image (layer before the classifier)",
                "Images": f"{len(features)} (up to {n} per class)",
                "Classes": ", ".join(class_names),
            },
            preview=figure_image(figure),
        )

    def _ask_params(self, parent: tk.Misc, sources: list[str]) -> tuple[str, int] | None:
        """Small dialog asking which features to use and how many images per class."""
        window = tk.Toplevel(parent)
        window.title(self.title)
        window.transient(parent)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Features of:").grid(row=0, column=0, sticky="w")
        source = ttk.Combobox(frame, values=sources, state="readonly", width=20)
        source.grid(row=0, column=1, padx=6, sticky="w")
        source.set(sources[0])

        ttk.Label(frame, text="Images per class (more = slower):").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        count = ttk.Spinbox(frame, from_=10, to=MAX_PER_CLASS, increment=50, width=8)
        count.grid(row=1, column=1, padx=6, pady=(6, 0), sticky="w")
        count.set(DEFAULT_PER_CLASS)

        result = None

        def apply() -> None:
            nonlocal result
            try:
                n = int(count.get())
            except ValueError:
                n = 0
            if not 10 <= n <= MAX_PER_CLASS:
                messagebox.showerror(
                    self.title, f"Type a number from 10 to {MAX_PER_CLASS}.", parent=window
                )
                return
            result = (source.get(), n)
            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(buttons, text="Project", command=apply).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)

        window.bind("<Return>", lambda _event: apply())
        window.bind("<Escape>", lambda _event: window.destroy())
        window.grab_set()
        window.wait_window()
        return result
