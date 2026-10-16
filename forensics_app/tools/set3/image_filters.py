
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult


class ImageFiltersTool(ForensicsTool):
    tool_id = "image_filters"
    title = "Image filters"
    category = "Set 3 · Filtering & Edges"
    description = "Apply different filters from scikit-image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:

        window = tk.Toplevel(parent)
        window.title("Image Filters")
        window.geometry("420x460")

        # Category dropdown
        ttk.Label(window, text="Choose a category:").pack(pady=10)

        categories = [
            "Smoothing / Noise Reduction",
            "Edge Detection",
            "Sharpening",
            "Thresholding"
        ]

        category_box = ttk.Combobox(
            window,
            values=categories,
            state="readonly",
            width=30
        )
        category_box.pack()
        category_box.current(0)

        # Filter dropdown
        filters_by_category = {
            "Smoothing / Noise Reduction": ["Gaussian", "Median"],
            "Edge Detection": ["Sobel", "Prewitt", "Scharr"],
            "Sharpening": ["Unsharp Mask"],
            "Thresholding": ["Otsu"]
        }

        ttk.Label(window, text="Choose a filter:").pack(pady=10)

        filter_box = ttk.Combobox(
            window,
            state="readonly",
            width=30
        )
        filter_box.pack()

        def update_filters(event=None):
            category = category_box.get()
            filter_box["values"] = filters_by_category[category]
            filter_box.current(0)

        update_filters()

        # Frame for adjustable settings
        settings_frame = ttk.Frame(window)
        settings_frame.pack(pady=15)

        # Gaussian settings
        gaussian_frame = ttk.Frame(settings_frame)

        ttk.Label(gaussian_frame, text="Sigma:").pack()

        sigma_entry = ttk.Entry(gaussian_frame, width=12)
        sigma_entry.pack()
        sigma_entry.insert(0, "2")

        # Median settings
        median_frame = ttk.Frame(settings_frame)

        ttk.Label(median_frame, text="Neighborhood size:").pack()

        size_entry = ttk.Entry(median_frame, width=12)
        size_entry.pack()
        size_entry.insert(0, "3")

        # Unsharp Mask settings
        unsharp_frame = ttk.Frame(settings_frame)

        ttk.Label(unsharp_frame, text="Radius:").pack()

        radius_entry = ttk.Entry(unsharp_frame, width=12)
        radius_entry.pack()
        radius_entry.insert(0, "1")

        ttk.Label(unsharp_frame, text="Amount:").pack(pady=(8, 0))

        amount_entry = ttk.Entry(unsharp_frame, width=12)
        amount_entry.pack()
        amount_entry.insert(0, "1")

        # Otsu settings
        otsu_frame = ttk.Frame(settings_frame)

        manual_threshold = tk.BooleanVar(value=False)

        threshold_entry = ttk.Entry(otsu_frame, width=12)
        threshold_entry.insert(0, "128")

        def update_threshold():
            if manual_threshold.get():
                threshold_entry.pack(pady=5)
            else:
                threshold_entry.pack_forget()

        ttk.Checkbutton(
            otsu_frame,
            text="Use manual threshold",
            variable=manual_threshold,
            command=update_threshold
        ).pack()

        # Show settings depending on the selected filter
        def update_settings(event=None):

            gaussian_frame.pack_forget()
            median_frame.pack_forget()
            unsharp_frame.pack_forget()
            otsu_frame.pack_forget()

            selected = filter_box.get()

            if selected == "Gaussian":
                gaussian_frame.pack()

            elif selected == "Median":
                median_frame.pack()

            elif selected == "Unsharp Mask":
                unsharp_frame.pack()

            elif selected == "Otsu":
                otsu_frame.pack()
                update_threshold()

        filter_box.bind("<<ComboboxSelected>>", update_settings)

        def category_changed(event=None):
            update_filters()
            update_settings()

        category_box.bind("<<ComboboxSelected>>", category_changed)

        update_settings()

        # Apply the selected filter
        result = None

        def apply_filter():
            nonlocal result

            try:
                selected = filter_box.get()

                image = np.array(document.current.convert("L")) / 255.0

                if selected == "Gaussian":
                    sigma = float(sigma_entry.get())

                    if not np.isfinite(sigma) or sigma <= 0:
                        raise ValueError("Sigma must be a positive number.")

                    filtered = filters.gaussian(image, sigma=sigma)

                elif selected == "Median":
                    size = int(size_entry.get())

                    if size < 1 or size > 15 or size % 2 == 0:
                        raise ValueError(
                            "Neighborhood size must be an odd number from 1 to 15."
                        )

                    filtered = filters.median(
                        image,
                        footprint=np.ones((size, size), dtype=bool)
                    )

                elif selected == "Sobel":
                    filtered = filters.sobel(image)

                elif selected == "Prewitt":
                    filtered = filters.prewitt(image)

                elif selected == "Scharr":
                    filtered = filters.scharr(image)

                elif selected == "Unsharp Mask":
                    radius = float(radius_entry.get())
                    amount = float(amount_entry.get())

                    if not np.isfinite(radius) or radius <= 0:
                        raise ValueError("Radius must be a positive number.")

                    if not np.isfinite(amount) or amount < 0:
                        raise ValueError("Amount must be zero or greater.")

                    filtered = filters.unsharp_mask(
                        image,
                        radius=radius,
                        amount=amount
                    )

                elif selected == "Otsu":

                    if manual_threshold.get():
                        threshold = float(threshold_entry.get())

                        if not np.isfinite(threshold) or not 0 <= threshold <= 255:
                            raise ValueError(
                                "Manual threshold must be between 0 and 255."
                            )

                        threshold = threshold / 255.0

                    else:
                        threshold = filters.threshold_otsu(image)

                    filtered = image > threshold

                else:
                    raise ValueError("Please select a filter.")

                # Convert the result back to a Pillow image
                filtered = np.clip(filtered * 255, 0, 255)
                result = Image.fromarray(filtered.astype("uint8"))

                window.destroy()

            except ValueError as error:
                messagebox.showerror(
                    "Invalid input",
                    str(error),
                    parent=window
                )

        # Apply and Cancel buttons
        ttk.Button(
            window,
            text="Apply",
            command=apply_filter
        ).pack(pady=10)

        ttk.Button(
            window,
            text="Cancel",
            command=window.destroy
        ).pack()

        window.grab_set()
        parent.wait_window(window)

        if result is None:
            return None

        return ToolResult(
            image=result,
            message="Image filter applied."
        )
