
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult


class CannyTool(ForensicsTool):
    tool_id = "canny"
    title = "Canny edge detection"
    category = "Set 3 · Filtering & Edges"
    description = "Detect edges using the Canny algorithm."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:

        window = tk.Toplevel(parent)
        window.title("Canny Edge Detection")
        window.geometry("400x420")

        # Sigma
        ttk.Label(window, text="Sigma:").pack(pady=10)

        sigma_entry = ttk.Entry(window, width=12)
        sigma_entry.pack()
        sigma_entry.insert(0, "2")

        # Optional thresholds
        custom_thresholds = tk.BooleanVar(value=False)

        threshold_frame = ttk.Frame(window)

        ttk.Label(
            threshold_frame,
            text="Low threshold:"
        ).pack()

        low_entry = ttk.Entry(threshold_frame, width=12)
        low_entry.pack()
        low_entry.insert(0, "0.1")

        ttk.Label(
            threshold_frame,
            text="High threshold:"
        ).pack(pady=(10, 0))

        high_entry = ttk.Entry(threshold_frame, width=12)
        high_entry.pack()
        high_entry.insert(0, "0.2")

        # Show or hide threshold inputs
        def update_thresholds():
            if custom_thresholds.get():
                threshold_frame.pack(pady=10)
            else:
                threshold_frame.pack_forget()

        ttk.Checkbutton(
            window,
            text="Customize thresholds",
            variable=custom_thresholds,
            command=update_thresholds
        ).pack(pady=10)

        # Apply Canny
        result = None

        def apply_canny():
            nonlocal result

            try:
                sigma = float(sigma_entry.get())

                if not np.isfinite(sigma) or sigma <= 0:
                    raise ValueError("Sigma must be a positive number.")

                # Convert image to grayscale and normalize it
                image = np.array(document.current.convert("L")) / 255.0

                if custom_thresholds.get():

                    low = float(low_entry.get())
                    high = float(high_entry.get())

                    if not np.isfinite(low) or not np.isfinite(high):
                        raise ValueError("Thresholds must be valid numbers.")

                    if not 0 <= low < high <= 1:
                        raise ValueError(
                            "Thresholds must be between 0 and 1, "
                            "and low must be smaller than high."
                        )

                    filtered = feature.canny(
                        image,
                        sigma=sigma,
                        low_threshold=low,
                        high_threshold=high
                    )

                else:
                    filtered = feature.canny(
                        image,
                        sigma=sigma
                    )

                # Convert True/False pixels to black/white
                result = Image.fromarray(
                    (filtered * 255).astype("uint8")
                )

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
            command=apply_canny
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
            message="Canny edge detection applied."
        )
