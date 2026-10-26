"""Set 5.4 - Look up the closest image in a folder.

Simplest algorithm (from the slide): the image with the most matches wins.
Better algorithm used here: RANSAC keeps only the matches that agree on one geometric
transformation (the inliers), so random wrong matches (outliers) no longer count.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import numpy as np
from skimage import measure, transform

from forensics_app.core import ImageDocument

from ..base import ForensicsTool, ToolResult
from .common import (
    ask_match_params,
    draw_matches,
    load_image,
    match_details,
    shrink,
    to_gray,
)
from .keypoints import extract_features
from .matching import match_features

DEFAULT_FOLDER = Path("images/set5-images")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_LISTED = 10  # how many images to list in the details panel

# A projective transformation (homography, as in the RANSAC slides) needs 4 point pairs.
# A match is an inlier if the transformation puts its point less than
# settings["ransac_threshold"] pixels away from its partner.
RANSAC_MIN_SAMPLES = 4


# Mark which matches are inliers (True) for the best projective transformation found by RANSAC
def find_inliers(
    keypoints1: np.ndarray, keypoints2: np.ndarray, matches: np.ndarray, settings: dict
) -> np.ndarray:
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
        transform.ProjectiveTransform,
        min_samples=RANSAC_MIN_SAMPLES,
        residual_threshold=settings["ransac_threshold"],
        max_trials=settings["ransac_trials"],
        rng=0,  # same result every time
    )
    return no_inliers if inliers is None else inliers


# Set5.4: Core function for lookup.
# Compare the reference with every image in the folder.
# Returns (path, inliers, matches) for each image, best first (most inliers).
def rank_folder(
    reference: np.ndarray, folder: Path, method: str, settings: dict, skip: Path | None
) -> list:
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


class ImageLookupTool(ForensicsTool):
    tool_id = "image_lookup"
    title = "Closest image in folder"
    category = "Set 5 · SIFT"
    description = "Find the image in a folder that best matches the current image (RANSAC inliers)."
    requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window

        default = str(DEFAULT_FOLDER) if DEFAULT_FOLDER.is_dir() else ""
        params = ask_match_params(
            parent, "Closest image in folder", pick_folder=True, default_path=default, ransac=True
        )
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
                message=(
                    "No other image with keypoints was found in the folder "
                    "(subfolders are not searched)."
                )
            )

        best_path, best_inliers, best_matches = scores[0]

        # With RANSAC, 0 inliers everywhere means no image really matches: say so instead of
        # picking one by its (unreliable) number of matches
        if settings["use_ransac"] and best_inliers == 0:
            details = {"Operation": "Closest image lookup", "Folder": folder}
            details.update(match_details(method, settings))
            details["Images compared"] = len(scores)
            for path, inlier_count, match_count in scores[:MAX_LISTED]:
                details[f"Inliers / matches: {path.name}"] = f"{inlier_count} / {match_count}"
            return ToolResult(
                message=(
                    "No reliable match found: no image has any RANSAC inliers. "
                    "Try SIFT, or turn RANSAC off to simply count matches."
                ),
                details=details,
            )

        # Match again with the best image and draw only the inlier lines
        best_image = load_image(best_path)
        ref_keypoints, ref_desc = extract_features(reference, method, settings)
        keypoints, desc = extract_features(to_gray(best_image), method, settings)
        matches = match_features(ref_desc, desc, method, settings)
        inliers = find_inliers(ref_keypoints, keypoints, matches, settings)
        output = draw_matches(
            reference_image, ref_keypoints, best_image, keypoints, matches[inliers]
        )

        details = {"Operation": "Closest image lookup", "Folder": folder}
        details.update(match_details(method, settings))
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
            message = (
                f"Closest image: {best_path.name} "
                f"({best_inliers} inliers out of {best_matches} matches)."
            )
        else:
            message = f"Closest image: {best_path.name} ({best_matches} matches)."
        return ToolResult(image=output, message=message, details=details)
