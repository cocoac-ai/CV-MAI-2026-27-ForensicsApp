"""Set 5.4 - Look up the closest image in a folder.

Simplest algorithm (from the slide): the image with the most matches wins.
Better algorithm used here: RANSAC keeps only the matches that agree on one geometric
transformation (the inliers), so random wrong matches (outliers) no longer count.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw
from skimage import feature, measure, transform

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult

METHOD_ORB = "ORB"
METHOD_SIFT = "SIFT"

DEFAULT_FOLDER = Path("images/set5-images")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_LISTED = 10  # how many images to list in the details panel
LINE_COLORS = ["red", "lime", "cyan", "yellow", "magenta", "orange", "deepskyblue", "violet"]

# Images are shrunk so the longest side is at most this many pixels. Then all images
# have a similar size (a big photo no longer gets many more matches) and SIFT runs faster.
MAX_SIDE = 800

# An affine transformation needs 3 point pairs. A match is an inlier if the transformation
# puts its point less than settings["ransac_threshold"] pixels away from its partner.
RANSAC_MIN_SAMPLES = 3


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


# Detect keypoints and extract their descriptors with ORB or SIFT
def extract_features(gray: np.ndarray, method: str, settings: dict) -> tuple[np.ndarray, np.ndarray]:
    if method == METHOD_ORB:
        extractor = feature.ORB(n_keypoints=settings["n_keypoints"])
    else:
        extractor = feature.SIFT(
            n_octaves=settings["n_octaves"],
            n_scales=settings["n_scales"],
            sigma_min=settings["sigma_min"],
        )
    extractor.detect_and_extract(gray)
    return extractor.keypoints, extractor.descriptors


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


# Mark which matches are inliers (True) for the best affine transformation found by RANSAC
def find_inliers(keypoints1: np.ndarray, keypoints2: np.ndarray, matches: np.ndarray, settings: dict) -> np.ndarray:
    if not settings["use_ransac"]:
        return np.ones(len(matches), dtype=bool)  # simplest algorithm: every match counts

    no_inliers = np.zeros(len(matches), dtype=bool)
    if len(matches) <= RANSAC_MIN_SAMPLES:
        return no_inliers  # not enough pairs to check anything

    # Keypoints are (row, col) but transformations use (x, y), so swap the columns
    points1 = keypoints1[matches[:, 0]][:, ::-1]
    points2 = keypoints2[matches[:, 1]][:, ::-1]
    _, inliers = measure.ransac(
        (points1, points2),
        transform.AffineTransform,
        min_samples=RANSAC_MIN_SAMPLES,
        residual_threshold=settings["ransac_threshold"],
        max_trials=settings["ransac_trials"],
        rng=0,  # same result every time
    )
    return no_inliers if inliers is None else inliers


# Set5: Core function for lookup.
# Compare the reference with every image in the folder.
# Returns (path, inliers, matches) for each image, best first (most inliers).
def rank_folder(reference: np.ndarray, folder: Path, method: str, settings: dict, skip: Path | None) -> list:
    ref_keypoints, ref_desc = extract_features(reference, method, settings)

    scores = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if skip is not None and path.resolve() == skip.resolve():
            continue  # the reference image itself would always win
        try:
            keypoints, desc = extract_features(to_gray(load_image(path)), method, settings)
        except (OSError, RuntimeError):
            continue  # not readable, or no keypoints found
        matches = match_features(ref_desc, desc, method, settings)
        inliers = find_inliers(ref_keypoints, keypoints, matches, settings)
        scores.append((path, int(inliers.sum()), len(matches)))

    # Sort by inliers; when two images have the same number, more matches wins
    scores.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return scores


# Put both images side by side and join every matched pair of keypoints with a line
def draw_matches(
    image1: Image.Image, keypoints1: np.ndarray, image2: Image.Image, keypoints2: np.ndarray, matches: np.ndarray
) -> Image.Image:
    height = max(image1.height, image2.height)
    offset = image1.width  # the second image starts after the first one
    output = Image.new("RGB", (offset + image2.width, height))
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


class ImageLookupTool(ForensicsTool):
    tool_id = "image_lookup"
    title = "Closest image in folder"
    category = "Set 5 · SIFT"
    description = "Find the image in a folder that best matches the current image (RANSAC inliers)."
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window

        params = self._ask_params(parent)
        if params is None:
            return None  # the user pressed Cancel
        folder, method, settings = params

        reference_image = shrink(document.current)
        reference = to_gray(reference_image)
        try:
            scores = rank_folder(reference, Path(folder), method, settings, skip=document.path)
        except RuntimeError as exc:
            return ToolResult(message=f"{method} failed on the current image: {exc}")
        if not scores:
            return ToolResult(
                message="No other image with keypoints was found in the folder (subfolders are not searched)."
            )

        best_path, best_inliers, best_matches = scores[0]

        # Match again with the best image and draw only the inlier lines
        best_image = load_image(best_path)
        ref_keypoints, ref_desc = extract_features(reference, method, settings)
        keypoints, desc = extract_features(to_gray(best_image), method, settings)
        matches = match_features(ref_desc, desc, method, settings)
        inliers = find_inliers(ref_keypoints, keypoints, matches, settings)
        output = draw_matches(reference_image, ref_keypoints, best_image, keypoints, matches[inliers])

        details = {"Operation": "Closest image lookup", "Folder": folder, "Method": method}
        if method == METHOD_ORB:
            details["Max keypoints"] = settings["n_keypoints"]
        else:
            details["Octaves"] = settings["n_octaves"]
            details["Scales per octave"] = settings["n_scales"]
            details["Sigma min"] = settings["sigma_min"]
        details["Max ratio"] = settings["max_ratio"]
        details["Cross check"] = settings["cross_check"]
        details["RANSAC"] = settings["use_ransac"]
        if settings["use_ransac"]:
            details["RANSAC threshold (pixels)"] = settings["ransac_threshold"]
            details["RANSAC trials"] = settings["ransac_trials"]
        details["Images compared"] = len(scores)
        details["Best image"] = best_path.name
        for path, inlier_count, match_count in scores[:MAX_LISTED]:
            if settings["use_ransac"]:
                details[f"Inliers / matches: {path.name}"] = f"{inlier_count} / {match_count}"
            else:
                details[f"Matches: {path.name}"] = match_count

        if settings["use_ransac"]:
            message = f"Closest image: {best_path.name} ({best_inliers} inliers out of {best_matches} matches)."
        else:
            message = f"Closest image: {best_path.name} ({best_matches} matches)."
        return ToolResult(image=output, message=message, details=details)

    def _ask_params(self, parent: tk.Misc) -> tuple[str, str, dict] | None:
        window = tk.Toplevel(parent)
        window.title("Closest image in folder")
        window.transient(parent)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)

        # Folder of images
        folder_box = ttk.LabelFrame(frame, text="Folder of images", padding=8)
        folder_box.pack(fill="both")
        folder_entry = ttk.Entry(folder_box, width=40)
        folder_entry.pack(side="left")
        if DEFAULT_FOLDER.is_dir():
            folder_entry.insert(0, str(DEFAULT_FOLDER))

        def browse() -> None:
            start = folder_entry.get().strip() or None  # start browsing from the folder in the box
            path = filedialog.askdirectory(parent=window, title="Choose a folder of images", initialdir=start)
            if path:
                folder_entry.delete(0, "end")
                folder_entry.insert(0, path)

        ttk.Button(folder_box, text="Browse…", command=browse).pack(side="left", padx=(6, 0))

        # Extractor
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
        match_fields = [("max_ratio", "Max ratio, 0 to 1 (lower = only clear matches):", "0.8")]
        entries = {}
        boxes = (("ORB parameters", orb_fields), ("SIFT parameters", sift_fields), ("Matching", match_fields))
        for box_title, fields in boxes:
            box = ttk.LabelFrame(frame, text=box_title, padding=8)
            box.pack(fill="both", pady=(8, 0))
            for key, label, default in fields:
                ttk.Label(box, text=label).pack(anchor="w")
                entry = ttk.Entry(box, width=10)
                entry.pack(anchor="w", pady=(4, 8))
                entry.insert(0, default)
                entries[key] = entry

        # Cross check goes in the last box ("Matching")
        cross_check = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Cross check (keep a match only if it is best both ways)", variable=cross_check).pack(
            anchor="w"
        )

        # RANSAC: keep only the matches that agree on one transformation
        ransac_box = ttk.LabelFrame(frame, text="RANSAC (remove outlier matches)", padding=8)
        ransac_box.pack(fill="both", pady=(8, 0))
        use_ransac = tk.BooleanVar(value=True)
        # Label and entry on the same row to keep the window short
        ttk.Checkbutton(
            ransac_box, text="Use RANSAC (off = simply count all matches)", variable=use_ransac
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(ransac_box, text="Inlier threshold in pixels:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        threshold_entry = ttk.Entry(ransac_box, width=8)
        threshold_entry.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        threshold_entry.insert(0, "5")
        ttk.Label(ransac_box, text="Number of trials:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        trials_entry = ttk.Entry(ransac_box, width=8)
        trials_entry.grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(4, 0))
        trials_entry.insert(0, "1000")

        result = None

        def apply() -> None:
            nonlocal result
            folder = folder_entry.get().strip()
            if not Path(folder).is_dir():
                messagebox.showerror("Invalid folder", "Please choose a folder.", parent=window)
                return
            try:
                settings = {
                    "n_keypoints": int(entries["n_keypoints"].get()),
                    "n_octaves": int(entries["n_octaves"].get()),
                    "n_scales": int(entries["n_scales"].get()),
                    "sigma_min": float(entries["sigma_min"].get()),
                    "max_ratio": float(entries["max_ratio"].get()),
                    "ransac_threshold": float(threshold_entry.get()),
                    "ransac_trials": int(trials_entry.get()),
                }
            except ValueError:
                settings = None
            if settings is None or min(settings.values()) <= 0 or settings["max_ratio"] > 1:
                messagebox.showerror(
                    "Invalid value",
                    "Please enter numbers above 0 (whole numbers, except sigma min, max ratio and threshold).\n"
                    "Max ratio must be between 0 and 1.",
                    parent=window,
                )
                return
            settings["cross_check"] = cross_check.get()
            settings["use_ransac"] = use_ransac.get()
            result = (folder, method.get(), settings)
            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.pack(pady=(12, 0))
        ttk.Button(buttons, text="Apply", command=apply).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)

        window.bind("<Return>", lambda _event: apply())
        window.bind("<Escape>", lambda _event: window.destroy())
        folder_entry.focus_set()
        window.grab_set()
        window.wait_window()
        return result
