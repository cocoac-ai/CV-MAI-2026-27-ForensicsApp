"""Set5 - Code shared by the Set 5 tools (5.1 corners, 5.2 keypoints, 5.3 matching, 5.4 lookup).

Each tool file keeps its own core function and what is special to it; the helpers used by more
than one tool are here.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw

from ..common import IMAGE_TYPES

METHOD_ORB = "ORB"
METHOD_SIFT = "SIFT"

# 5.3 and 5.4 shrink images so the longest side is at most this many pixels. Then both images
# have a similar size (a big photo no longer gets many more matches) and SIFT runs faster.
MAX_SIDE = 800
LINE_COLORS = ["red", "lime", "cyan", "yellow", "magenta", "orange", "deepskyblue", "violet"]


# ---------------------------------------------------------------------------
# Drawing (5.1, 5.2)
# ---------------------------------------------------------------------------

# Draw a red circle around every point on the original (colour) image, like plt.scatter in the slide
def draw_points(image: Image.Image, points: np.ndarray) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)

    # Big images are shrunk to fit the window, so their circles must be bigger to stay visible
    radius = max(4, max(output.size) // 250)
    width = max(1, radius // 4)
    for row, col in points:
        draw.ellipse(
            (col - radius, row - radius, col + radius, row + radius), outline="red", width=width
        )
    return output


# ---------------------------------------------------------------------------
# Loading images (5.3, 5.4)
# ---------------------------------------------------------------------------

# Colour copy of an image, shrunk to MAX_SIDE (used for drawing)
def shrink(image: Image.Image) -> Image.Image:
    small = image.convert("RGB")
    small.thumbnail((MAX_SIDE, MAX_SIDE))  # keeps the aspect ratio, never enlarges
    return small


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return shrink(image)


# ORB and SIFT only work on one channel, so features are computed on the grayscale version
def to_gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"))


# ---------------------------------------------------------------------------
# Drawing matches (5.3, 5.4)
# ---------------------------------------------------------------------------

# Put both images side by side and join every matched pair of keypoints with a line
def draw_matches(
    image1: Image.Image,
    keypoints1: np.ndarray,
    image2: Image.Image,
    keypoints2: np.ndarray,
    matches: np.ndarray,
) -> Image.Image:
    height = max(image1.height, image2.height)
    offset = image1.width  # the second image starts after the first one
    # White background, so the space under the shorter image is not a black bar
    output = Image.new("RGB", (offset + image2.width, height), "white")
    output.paste(image1, (0, 0))
    output.paste(image2, (offset, 0))

    draw = ImageDraw.Draw(output)
    for number, (index1, index2) in enumerate(matches):
        row1, col1 = keypoints1[index1]
        row2, col2 = keypoints2[index2]
        col2 += offset
        color = LINE_COLORS[number % len(LINE_COLORS)]
        draw.line((col1, row1, col2, row2), fill=color, width=1)
        draw.ellipse((col1 - 3, row1 - 3, col1 + 3, row1 + 3), outline=color)
        draw.ellipse((col2 - 3, row2 - 3, col2 + 3, row2 + 3), outline=color)
    return output


# ---------------------------------------------------------------------------
# Parameter windows (5.2, 5.3, 5.4)
# ---------------------------------------------------------------------------

# Add the ORB/SIFT choice and their parameter boxes to a window
def add_extractor_options(frame: ttk.Frame) -> tuple[tk.StringVar, dict]:
    method_box = ttk.LabelFrame(frame, text="Descriptor extractor", padding=8)
    method_box.pack(fill="both", pady=(8, 0))
    method = tk.StringVar(value=METHOD_ORB)
    for name in (METHOD_ORB, METHOD_SIFT):
        ttk.Radiobutton(method_box, text=name, value=name, variable=method).pack(anchor="w")

    # (key, label, default) for each parameter box
    orb_fields = [("n_keypoints", "Max number of keypoints:", "200")]
    sift_fields = [
        ("n_octaves", "Number of octaves:", "8"),
        ("n_scales", "Scales per octave:", "3"),
        ("sigma_min", "Sigma min (blur of the first scale):", "1.6"),
    ]
    entries = {}
    for box_title, fields in (("ORB parameters", orb_fields), ("SIFT parameters", sift_fields)):
        box = ttk.LabelFrame(frame, text=box_title, padding=8)
        box.pack(fill="both", pady=(8, 0))
        for key, label, default in fields:
            ttk.Label(box, text=label).pack(anchor="w")
            entry = ttk.Entry(box, width=10)
            entry.pack(anchor="w", pady=(4, 8))
            entry.insert(0, default)
            entries[key] = entry
    return method, entries


# Read the values typed in the boxes above; None if one of them is not valid
def read_extractor_options(entries: dict) -> dict | None:
    try:
        settings = {
            "n_keypoints": int(entries["n_keypoints"].get()),
            "n_octaves": int(entries["n_octaves"].get()),
            "n_scales": int(entries["n_scales"].get()),
            "sigma_min": float(entries["sigma_min"].get()),
        }
    except ValueError:
        return None
    if min(settings.values()) <= 0:
        return None
    return settings


# Parameter window for 5.3 and 5.4.
# pick_folder=True asks for a folder instead of a file; ransac=True adds the RANSAC box (5.4 only)
def ask_match_params(
    parent: tk.Misc, title: str, pick_folder: bool, default_path: str = "", ransac: bool = False
) -> tuple[str, str, dict] | None:
    window = tk.Toplevel(parent)
    window.title(title)
    window.transient(parent)
    window.resizable(False, False)

    frame = ttk.Frame(window, padding=12)
    frame.pack(fill="both", expand=True)

    # Second image or folder
    path_box = ttk.LabelFrame(
        frame, text="Folder of images" if pick_folder else "Second image", padding=8
    )
    path_box.pack(fill="both")
    path_entry = ttk.Entry(path_box, width=40)
    path_entry.pack(side="left")
    path_entry.insert(0, default_path)

    def browse() -> None:
        start = path_entry.get().strip() or None  # start browsing from the path in the box
        if pick_folder:
            path = filedialog.askdirectory(
                parent=window, title="Choose a folder of images", initialdir=start
            )
        else:
            path = filedialog.askopenfilename(
                parent=window, title="Choose a second image", filetypes=IMAGE_TYPES
            )
        if path:
            path_entry.delete(0, "end")
            path_entry.insert(0, path)

    ttk.Button(path_box, text="Browse…", command=browse).pack(side="left", padx=(6, 0))

    # ORB / SIFT and their parameters
    method, entries = add_extractor_options(frame)

    # Matching parameters
    match_box = ttk.LabelFrame(frame, text="Matching", padding=8)
    match_box.pack(fill="both", pady=(8, 0))
    ttk.Label(match_box, text="Max ratio, 0 to 1 (lower = only clear matches):").pack(anchor="w")
    ratio_entry = ttk.Entry(match_box, width=10)
    ratio_entry.pack(anchor="w", pady=(4, 8))
    ratio_entry.insert(0, "0.8")
    cross_check = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        match_box,
        text="Cross check (keep a match only if it is best both ways)",
        variable=cross_check,
    ).pack(anchor="w")

    # RANSAC parameters: keep only the matches that agree on one transformation (5.4 only)
    use_ransac = tk.BooleanVar(value=True)
    if ransac:
        ransac_box = ttk.LabelFrame(frame, text="RANSAC (remove outlier matches)", padding=8)
        ransac_box.pack(fill="both", pady=(8, 0))
        # Label and entry on the same row to keep the window short
        ttk.Checkbutton(
            ransac_box, text="Use RANSAC (off = simply count all matches)", variable=use_ransac
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(ransac_box, text="Inlier threshold in pixels:").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        threshold_entry = ttk.Entry(ransac_box, width=8)
        threshold_entry.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        threshold_entry.insert(0, "5")
        ttk.Label(ransac_box, text="Number of trials:").grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )
        trials_entry = ttk.Entry(ransac_box, width=8)
        trials_entry.grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(4, 0))
        trials_entry.insert(0, "1000")

    result = None

    def apply() -> None:
        nonlocal result
        path = path_entry.get().strip()
        path_ok = Path(path).is_dir() if pick_folder else Path(path).is_file()
        if not path_ok:
            what = "a folder" if pick_folder else "a second image file"
            messagebox.showerror("Invalid path", f"Please choose {what}.", parent=window)
            return
        settings = read_extractor_options(entries)
        if settings is None:
            messagebox.showerror(
                "Invalid value",
                "Please enter numbers above 0 (whole numbers, except sigma min).",
                parent=window,
            )
            return
        try:
            max_ratio = float(ratio_entry.get())
        except ValueError:
            max_ratio = -1.0
        if not 0 < max_ratio <= 1:
            messagebox.showerror(
                "Invalid ratio", "Max ratio must be between 0 and 1.", parent=window
            )
            return
        settings["max_ratio"] = max_ratio
        settings["cross_check"] = cross_check.get()
        if ransac:
            try:
                settings["ransac_threshold"] = float(threshold_entry.get())
                settings["ransac_trials"] = int(trials_entry.get())
            except ValueError:
                settings["ransac_threshold"] = settings["ransac_trials"] = 0
            if settings["ransac_threshold"] <= 0 or settings["ransac_trials"] < 1:
                messagebox.showerror(
                    "Invalid RANSAC value",
                    "Threshold must be above 0 and trials a whole number of 1 or more.",
                    parent=window,
                )
                return
            settings["use_ransac"] = use_ransac.get()
        result = (path, method.get(), settings)
        window.destroy()

    buttons = ttk.Frame(frame)
    buttons.pack(pady=(12, 0))
    ttk.Button(buttons, text="Apply", command=apply).pack(side="left", padx=4)
    ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)

    window.bind("<Return>", lambda _event: apply())
    window.bind("<Escape>", lambda _event: window.destroy())
    path_entry.focus_set()
    window.grab_set()
    window.wait_window()
    return result


# ---------------------------------------------------------------------------
# Rows for the details panel (5.2, 5.3, 5.4)
# ---------------------------------------------------------------------------

# Only the parameters of the chosen method
def extractor_details(method: str, settings: dict) -> dict:
    if method == METHOD_ORB:
        return {"Method": method, "Max keypoints": settings["n_keypoints"]}
    return {
        "Method": method,
        "Octaves": settings["n_octaves"],
        "Scales per octave": settings["n_scales"],
        "Sigma min": settings["sigma_min"],
    }


def match_details(method: str, settings: dict) -> dict:
    details = extractor_details(method, settings)
    details["Max ratio"] = settings["max_ratio"]
    details["Cross check"] = settings["cross_check"]
    return details
