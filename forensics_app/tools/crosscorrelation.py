
"""Interactive normalized cross-correlation tool."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class NormalizedCrossCorrelationTool(ForensicsTool):
    tool_id = "normalized_cross_correlation"
    title = "Normalized cross-correlation"
    category = "Template matching"
    description = "Find a template using normalized cross-correlation."

    # This tool does not require an image to be loaded first
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:

        # Images are selected inside this tool
        selected = {
            "image": None,
            "template": None
        }

        result = {
            "image": None,
            "max_corr": None
        }

        # Create the tool window
        window = tk.Toplevel(parent)
        window.title("Normalized Cross-Correlation")
        window.geometry("1100x750")

        image_label = tk.StringVar(value="No original image selected")
        template_label = tk.StringVar(value="No template selected")
        threshold = tk.StringVar(value="0.8")

        # Choose original image
        def choose_image():
            path = filedialog.askopenfilename(
                parent=window,
                title="Select Original Image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp")
                ]
            )

            if path:
                try:
                    with Image.open(path) as img:
                        selected["image"] = img.copy()
                    image_label.set(path)
                except Exception as exc:
                    messagebox.showerror(
                        "Image error", str(exc), parent=window
                    )

        # Choose template
        def choose_template():
            path = filedialog.askopenfilename(
                parent=window,
                title="Select Template",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp")
                ]
            )

            if path:
                try:
                    with Image.open(path) as img:
                        selected["template"] = img.copy()
                    template_label.set(path)
                except Exception as exc:
                    messagebox.showerror(
                        "Template error", str(exc), parent=window
                    )

        # Controls
        controls = ttk.Frame(window, padding=10)
        controls.pack(fill="x")

        ttk.Button(
            controls,
            text="Choose Original Image",
            command=choose_image
        ).pack(anchor="w")

        ttk.Label(
            controls,
            textvariable=image_label
        ).pack(anchor="w")

        ttk.Button(
            controls,
            text="Choose Template",
            command=choose_template
        ).pack(anchor="w", pady=(10, 0))

        ttk.Label(
            controls,
            textvariable=template_label
        ).pack(anchor="w")

        ttk.Label(
            controls,
            text="Correlation threshold (0 to 1)"
        ).pack(anchor="w", pady=(10, 0))

        ttk.Entry(
            controls,
            textvariable=threshold,
            width=10
        ).pack(anchor="w")

        # Create plots
        fig = Figure(figsize=(12, 6))
        axes = fig.subplots(1, 4)

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # Run cross-correlation
        def run_correlation():

            if selected["image"] is None:
                messagebox.showwarning(
                    "Missing image",
                    "Please select an original image.",
                    parent=window
                )
                return

            if selected["template"] is None:
                messagebox.showwarning(
                    "Missing template",
                    "Please select a template.",
                    parent=window
                )
                return

            try:
                t = float(threshold.get())
            except ValueError:
                messagebox.showerror(
                    "Invalid threshold",
                    "Enter a number between 0 and 1.",
                    parent=window
                )
                return

            if not np.isfinite(t) or not 0 <= t <= 1:
                messagebox.showerror(
                    "Invalid threshold",
                    "Threshold must be between 0 and 1.",
                    parent=window
                )
                return

            # Convert images to grayscale arrays
            image = np.asarray(
                ImageOps.grayscale(selected["image"]),
                dtype=float
            )

            template = np.asarray(
                ImageOps.grayscale(selected["template"]),
                dtype=float
            )

            # Validate template size
            if (template.shape[0] > image.shape[0] or
                    template.shape[1] > image.shape[1]):
                messagebox.showerror(
                    "Invalid template",
                    "Template must fit inside the original image.",
                    parent=window
                )
                return

            if np.std(template) == 0:
                messagebox.showerror(
                    "Invalid template",
                    "Template must contain intensity variation.",
                    parent=window
                )
                return

            # Normalized cross-correlation
            corr = feature.match_template(
                image,
                template,
                pad_input=True
            )

            # Apply threshold
            matches = corr >= t
            y, x = np.where(matches)

            # Clear old plots
            for ax in axes:
                ax.clear()

            # Original image
            axes[0].imshow(image, cmap="gray")
            axes[0].set_title("Original Image")

            # Template
            axes[1].imshow(template, cmap="gray")
            axes[1].set_title("Template")

            # Correlation map
            axes[2].imshow(
                corr,
                cmap="gray",
                vmin=-1,
                vmax=1
            )
            axes[2].set_title("Normalized Cross-Correlation")

            # Thresholded image
            axes[3].imshow(
                image * 0.4,
                cmap="gray",
                vmin=0,
                vmax=255
            )
            axes[3].plot(
                x, y, "r.", markersize=2
            )
            axes[3].set_title(
                f"Thresholded Image ({t})"
            )

            for ax in axes:
                ax.axis("off")

            fig.tight_layout()
            canvas.draw()

            # Convert correlation map to PIL image
            corr_image = (
                (np.clip(corr, -1, 1) + 1) * 127.5
            ).astype(np.uint8)

            result["image"] = Image.fromarray(corr_image)
            result["max_corr"] = float(np.max(corr))

        ttk.Button(
            controls,
            text="Run Cross-Correlation",
            command=run_correlation
        ).pack(anchor="w", pady=10)

        ttk.Button(
            controls,
            text="Done",
            command=window.destroy
        ).pack(anchor="w")

        window.protocol("WM_DELETE_WINDOW", window.destroy)

        # Wait for the user to finish
        window.wait_window()

        # Return the result
        return ToolResult(
            image=result["image"],
            message=(
                "Normalized cross-correlation completed."
                if result["image"] is not None
                else "No cross-correlation was performed."
            ),
            details={
                "Operation": "Normalized cross-correlation",
                "Maximum correlation": result["max_corr"]
            }
        )
