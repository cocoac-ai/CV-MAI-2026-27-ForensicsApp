
"""Set 6.2 — Eigenfaces (PCA) using the LFW face dataset."""

import csv
import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from ..base import ForensicsTool, ToolResult
from ..common import PROJECT_FOLDER


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

# Everything this version makes goes in one folder (ignored by git): the LFW download
# (lfw_home/), the face PNGs (train/, test/) and the model. Deleting it removes all of it.
DATA_DIR = PROJECT_FOLDER / "images" / "set6-eigenfaces"
PCA_FILE = DATA_DIR / "eigenfaces_pca.npz"

N_COMPONENTS = 100
RANDOM_STATE = 42


# ---------------------------------------------------------
# 1. Load the LFW dataset
# ---------------------------------------------------------

def load_lfw_data():
    """Load LFW and split faces into training and test sets."""

    lfw = fetch_lfw_people(
        data_home=DATA_DIR,
        min_faces_per_person=20,
        resize=0.4,
    )

    # X contains flattened grayscale images.
    # y contains the corresponding person IDs.
    X = lfw.data.astype(np.float32)
    y = lfw.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return lfw, X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# 2. Save face images and person labels
# ---------------------------------------------------------

def save_face_images(X, y, target_names, image_shape, folder):
    """
    Save face images as PNG files.

    labels.csv preserves the correspondence between
    each image and the person's name.
    """

    folder.mkdir(parents=True, exist_ok=True)

    labels_file = folder / "labels.csv"

    with open(
        labels_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "person_name",
        ])

        for i, (face, label) in enumerate(zip(X, y)):

            filename = f"face_{i:04d}.png"

            # Convert flattened face back to 2D.
            image_array = face.reshape(image_shape)

            # LFW pixels are 0..1, so scale them to 0..255 first.
            image_array = np.clip(
                image_array * 255,
                0,
                255,
            ).astype(np.uint8)

            image = Image.fromarray(image_array)

            image.save(folder / filename)

            person_name = str(target_names[int(label)])

            writer.writerow([
                filename,
                person_name,
            ])


# ---------------------------------------------------------
# 3. Train PCA and calculate eigenfaces
# ---------------------------------------------------------

def train_eigenfaces(n_components=N_COMPONENTS):
    """
    Train PCA using training faces only.

    Save the mean face, eigenfaces, PCA coefficients,
    training/test images, and labels.
    """

    lfw, X_train, X_test, y_train, y_test = load_lfw_data()

    image_shape = lfw.images.shape[1:]

    # Save training images and their labels.
    save_face_images(
        X_train,
        y_train,
        lfw.target_names,
        image_shape,
        DATA_DIR / "train",
    )

    # Save test images and their labels.
    save_face_images(
        X_test,
        y_test,
        lfw.target_names,
        image_shape,
        DATA_DIR / "test",
    )

    # PCA cannot use more components than the
    # number of training samples or image features.
    n_components = min(
        n_components,
        X_train.shape[0],
        X_train.shape[1],
    )

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        whiten=False,
        random_state=RANDOM_STATE,
    )

    # Fit PCA ONLY on training images.
    train_coefficients = pca.fit_transform(X_train)

    # Average face of the training dataset.
    mean_face = pca.mean_

    # Principal components: the eigenfaces.
    eigenfaces = pca.components_

    # Project test images without retraining PCA.
    test_coefficients = pca.transform(X_test)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save everything needed for Set 6.3.
    np.savez_compressed(
        PCA_FILE,

        mean_face=mean_face,
        eigenfaces=eigenfaces,

        train_coefficients=train_coefficients,
        test_coefficients=test_coefficients,

        X_train=X_train,
        X_test=X_test,

        y_train=y_train,
        y_test=y_test,

        target_names=lfw.target_names,

        image_shape=np.array(image_shape),

        explained_variance_ratio=(
            pca.explained_variance_ratio_
        ),
    )

    return {
        "mean_face": mean_face,
        "eigenfaces": eigenfaces,
        "train_coefficients": train_coefficients,
        "test_coefficients": test_coefficients,
        "image_shape": image_shape,
        "training_count": len(X_train),
        "test_count": len(X_test),
        "person_count": len(lfw.target_names),
        "explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
    }


# ---------------------------------------------------------
# 4. Convert face vectors into displayable images
# ---------------------------------------------------------

def face_to_image(face, image_shape):
    """
    Convert a flattened face into a PIL image.

    Normalize each image for visualization.
    """

    array = np.asarray(
        face,
        dtype=np.float32,
    ).reshape(image_shape)

    minimum = float(array.min())
    maximum = float(array.max())

    if maximum > minimum:
        array = (
            (array - minimum)
            / (maximum - minimum)
        )
    else:
        array = np.zeros_like(array)

    array = (array * 255).astype(np.uint8)

    return Image.fromarray(
        array,
        mode="L",
    ).convert("RGB")


# ---------------------------------------------------------
# 5. Visualize the first K eigenfaces
# ---------------------------------------------------------

def visualize_eigenfaces(result, k=10):
    """
    Create a grid containing:
    - Mean face
    - First K eigenfaces
    """

    mean_face = result["mean_face"]
    eigenfaces = result["eigenfaces"]
    image_shape = result["image_shape"]

    k = min(k, len(eigenfaces))

    columns = 5
    tile_width = 160
    tile_height = 190

    count = k + 1

    rows = (
        count + columns - 1
    ) // columns

    canvas = Image.new(
        "RGB",
        (
            columns * tile_width,
            rows * tile_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    faces = [
        mean_face,
        *eigenfaces[:k],
    ]

    labels = [
        "Mean face",
        *[
            f"Eigenface {i + 1}"
            for i in range(k)
        ],
    ]

    for index, (face, label) in enumerate(
        zip(faces, labels)
    ):

        row, column = divmod(
            index,
            columns,
        )

        image = face_to_image(
            face,
            image_shape,
        )

        # Keep the face's width/height ratio.
        image = ImageOps.contain(
            image,
            (140, 145),
        )

        x = column * tile_width + 10
        y = row * tile_height + 10

        canvas.paste(
            image,
            (x, y),
        )

        draw.text(
            (x, y + 150),
            label,
            fill="black",
        )

    return canvas


# ---------------------------------------------------------
# 6. Connect Eigenfaces to the Forensics App
# ---------------------------------------------------------

class EigenfacesTool(ForensicsTool):

    tool_id = "eigenfaces"

    title = "Eigenfaces (PCA)"

    category = "Set 6 · Faces"

    description = (
        "Train PCA on LFW and visualize "
        "the mean face and eigenfaces."
    )

    requires_image = False

    def run(
        self,
        parent: tk.Misc,
        document,
    ) -> ToolResult | None:

        k = simpledialog.askinteger(
            "Eigenfaces",
            "How many eigenfaces would you like to display?",
            initialvalue=10,
            minvalue=1,
            maxvalue=N_COMPONENTS,
            parent=parent,
        )

        if k is None:
            return None

        try:

            result = train_eigenfaces()

            output = visualize_eigenfaces(
                result,
                k,
            )

        except Exception as exc:

            messagebox.showerror(
                "Eigenfaces error",
                f"Could not train Eigenfaces:\n\n{exc}",
                parent=parent,
            )

            return None

        # preview: show the grid without needing an open image.
        return ToolResult(
            preview=output,

            message=(
                "Eigenfaces trained successfully. "
                f"{result['training_count']} training images, "
                f"{result['test_count']} test images."
            ),

            details={
                "Dataset": "LFW",
                "PCA components": len(
                    result["eigenfaces"]
                ),
                "Displayed eigenfaces": k,
                "Training images": result[
                    "training_count"
                ],
                "Test images": result[
                    "test_count"
                ],
                "People": result[
                    "person_count"
                ],
                "Explained variance": (
                    f"{result['explained_variance']:.2%}"
                ),
                "Model saved": str(PCA_FILE),
                "Training folder": str(
                    DATA_DIR / "train"
                ),
                "Test folder": str(
                    DATA_DIR / "test"
                ),
            },
        )
