"""Set 4.4 - Person detection using HOG and a sliding window.

"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

# HOG parameters from the slide
ORIENTATIONS = 16
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)

METHOD_EUCLIDEAN = "Euclidean distance"
METHOD_DOT = "Dot product"

DEFAULT_TEMPLATE = Path("images/set4-images/person_template.bmp")
MAX_WINDOWS = 20000  # more windows than this would freeze the app for too long


# HOG feature vector of an RGB image (same call as the slide)
def hog_features(image: np.ndarray) -> np.ndarray:
    return feature.hog(
        image,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        channel_axis=-1,
    )


# Set4: Core fucntion
# Slide a template-sized window over the image and compare HOG features
def find_person(image: np.ndarray, template: np.ndarray, step: int) -> dict:
    window_h, window_w = template.shape[:2]
    image_h, image_w = image.shape[:2]
    tmpl_fd = hog_features(template)

    euclidean_differences = []
    dot_prod_differences = []
    coord = []

    # +1 so the last possible window position is also checked
    for i in range(0, image_h - window_h + 1, step):
        for j in range(0, image_w - window_w + 1, step):
            coord.append((i, j))

            person_slice = image[i:i + window_h, j:j + window_w]
            img_fd = hog_features(person_slice)

            euclidean_differences.append(np.linalg.norm(tmpl_fd - img_fd))
            dot_prod_differences.append(np.dot(img_fd, tmpl_fd))

    min_euclid_location = int(np.argmin(euclidean_differences))
    max_dot_prod_location = int(np.argmax(dot_prod_differences))

    return {
        "euclidean_position": coord[min_euclid_location],
        "euclidean_value": float(euclidean_differences[min_euclid_location]),
        "dot_position": coord[max_dot_prod_location],
        "dot_value": float(dot_prod_differences[max_dot_prod_location]),
        "windows": len(coord),
    }


# Draw a red box of the template size at (row, col)
def draw_box(image: Image.Image, position: tuple[int, int], size: tuple[int, int]) -> Image.Image:
    row, col = position
    height, width = size
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    draw.rectangle((col, row, col + width - 1, row + height - 1), outline="red", width=2)
    return output


def count_windows(image_size: tuple[int, int], window_size: tuple[int, int], step: int) -> int:
    rows = (image_size[0] - window_size[0]) // step + 1
    cols = (image_size[1] - window_size[1]) // step + 1
    return rows * cols


class PersonDetectionTool(ForensicsTool):
    tool_id = "person_detection"
    title = "Person detection (HOG)"
    category = "Set4"
    description = "Find the window most similar to a person template using HOG."
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window

        params = self._ask_params(parent)
        if params is None:
            return None  # the user pressed Cancel
        template_path, step, method = params

        # Both images as RGB, so their HOG vectors have the same length
        image = np.asarray(document.current.convert("RGB"))
        with Image.open(template_path) as template_file:
            template = np.asarray(template_file.convert("RGB"))

        window_size = template.shape[:2]
        if image.shape[0] < window_size[0] or image.shape[1] < window_size[1]:
            return ToolResult(
                message=(
                    f"The image is smaller than the template "
                    f"({window_size[1]}x{window_size[0]} pixels)."
                )
            )

        windows = count_windows(image.shape[:2], window_size, step)
        if windows > MAX_WINDOWS:
            return ToolResult(
                message=f"Too many windows ({windows}). Use a larger step or a smaller image."
            )

        found = find_person(image, template, step)

        if method == METHOD_EUCLIDEAN:
            position, value = found["euclidean_position"], found["euclidean_value"]
        else:
            position, value = found["dot_position"], found["dot_value"]

        output = draw_box(document.current, position, window_size)

        return ToolResult(
            image=output,
            message=f"Person found with {method.lower()} at row {position[0]}, column {position[1]}.",
            details={
                "Operation": "Person detection (HOG)",
                "Template": Path(template_path).name,
                "Window size": f"{window_size[1]} x {window_size[0]}",
                "Step": step,
                "Windows checked": found["windows"],
                "Method": method,
                "Best position (row, col)": position,
                "Score": f"{value:.3f}",
                "Euclidean best (row, col)": found["euclidean_position"],
                "Dot product best (row, col)": found["dot_position"],
            },
        )

    def _ask_params(self, parent: tk.Misc) -> tuple[str, int, str] | None:
        window = tk.Toplevel(parent)
        window.title("Person detection (HOG)")
        window.transient(parent)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)

        # Template file
        template_box = ttk.LabelFrame(frame, text="Person template", padding=8)
        template_box.pack(fill="both")
        template_entry = ttk.Entry(template_box, width=40)
        template_entry.pack(side="left")
        if DEFAULT_TEMPLATE.exists():
            template_entry.insert(0, str(DEFAULT_TEMPLATE))

        def browse() -> None:
            path = filedialog.askopenfilename(
                parent=window,
                title="Choose a person template",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*")],
            )
            if path:
                template_entry.delete(0, "end")
                template_entry.insert(0, path)

        ttk.Button(template_box, text="Browse…", command=browse).pack(side="left", padx=(6, 0))

        # Sliding window step
        step_box = ttk.LabelFrame(frame, text="Sliding window", padding=8)
        step_box.pack(fill="both", pady=(8, 0))
        ttk.Label(step_box, text="Step size in pixels (smaller = slower but more precise):").pack(anchor="w")
        step_entry = ttk.Entry(step_box, width=10)
        step_entry.pack(anchor="w", pady=(4, 0))
        step_entry.insert(0, "4")

        # Comparison method
        method_box = ttk.LabelFrame(frame, text="Compare HOG features with", padding=8)
        method_box.pack(fill="both", pady=(8, 0))
        method = tk.StringVar(value=METHOD_EUCLIDEAN)
        for name in (METHOD_EUCLIDEAN, METHOD_DOT):
            ttk.Radiobutton(method_box, text=name, value=name, variable=method).pack(anchor="w")

        result = None

        def apply() -> None:
            nonlocal result
            path = template_entry.get().strip()
            if not Path(path).is_file():
                messagebox.showerror("Invalid template", "Please choose a template image file.", parent=window)
                return
            try:
                step = int(step_entry.get())
            except ValueError:
                step = 0
            if step < 1:
                messagebox.showerror("Invalid step", "Step must be a whole number of 1 or more.", parent=window)
                return
            result = (path, step, method.get())
            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.pack(pady=(12, 0))
        ttk.Button(buttons, text="Apply", command=apply).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)

        window.bind("<Return>", lambda _event: apply())
        window.bind("<Escape>", lambda _event: window.destroy())
        step_entry.focus_set()
        window.grab_set()
        window.wait_window()
        return result
