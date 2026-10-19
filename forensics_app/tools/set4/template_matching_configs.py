
"""Exercise 4.2: Configurable template matching with Canny superimposition."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps, ImageDraw

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from skimage import feature, transform

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult


class TemplateMatchingConfigsTool(ForensicsTool):

    tool_id = "template_matching_configs"
    title = "Template Matching Configs"
    category = "Set 4 · HOG & Detection"
    description = (
        "Template matching with configurable rotation, "
        "threshold, and Canny edge superimposition."
    )

    requires_image = False

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument
    ) -> ToolResult | None:

        selected = {
            "image": None,
            "template": None
        }

        result = {
            "image": None,
            "score": None,
            "angle": None,
            "threshold": None,
            "matches": 0
        }

        # ---------------------------------------
        # Window
        # ---------------------------------------

        window = tk.Toplevel(parent)
        window.title("Template Matching Configs")
        window.geometry("1200x750")
        window.minsize(850, 550)

        image_label = tk.StringVar(
            value="No original image selected"
        )

        template_label = tk.StringVar(
            value="No template selected"
        )

        angle_var = tk.StringVar(value="0")
        threshold_var = tk.StringVar(value="0.8")
        sigma_var = tk.StringVar(value="2")

        status_var = tk.StringVar(
            value="Choose an image and template, then click Run."
        )

        # ---------------------------------------
        # Image selection
        # ---------------------------------------

        def choose_image():

            path = filedialog.askopenfilename(
                parent=window,
                title="Choose Original Image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                    ("All files", "*.*")
                ]
            )

            if not path:
                return

            try:
                with Image.open(path) as img:
                    selected["image"] = img.convert("RGB")

                image_label.set(path)

            except Exception as error:
                messagebox.showerror(
                    "Image Error",
                    str(error),
                    parent=window
                )

        def choose_template():

            path = filedialog.askopenfilename(
                parent=window,
                title="Choose Template",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                    ("All files", "*.*")
                ]
            )

            if not path:
                return

            try:
                with Image.open(path) as img:
                    selected["template"] = img.convert("RGB")

                template_label.set(path)

            except Exception as error:
                messagebox.showerror(
                    "Template Error",
                    str(error),
                    parent=window
                )

        # ---------------------------------------
        # Top controls
        # ---------------------------------------

        controls = ttk.Frame(window, padding=10)
        controls.pack(side="top", fill="x")

        ttk.Button(
            controls,
            text="Choose Original Image",
            command=choose_image
        ).pack(anchor="w")

        ttk.Label(
            controls,
            textvariable=image_label,
            wraplength=1050
        ).pack(anchor="w", pady=(2, 8))

        ttk.Button(
            controls,
            text="Choose Template",
            command=choose_template
        ).pack(anchor="w")

        ttk.Label(
            controls,
            textvariable=template_label,
            wraplength=1050
        ).pack(anchor="w", pady=(2, 8))

        settings = ttk.Frame(controls)
        settings.pack(fill="x", pady=8)

        ttk.Label(
            settings,
            text="Rotation Angle (degrees):"
        ).grid(row=0, column=0, padx=5, sticky="w")

        ttk.Entry(
            settings,
            textvariable=angle_var,
            width=10
        ).grid(row=0, column=1, padx=5)

        ttk.Label(
            settings,
            text="Correlation Threshold:"
        ).grid(row=0, column=2, padx=5, sticky="w")

        ttk.Entry(
            settings,
            textvariable=threshold_var,
            width=10
        ).grid(row=0, column=3, padx=5)

        ttk.Label(
            settings,
            text="Canny Sigma:"
        ).grid(row=0, column=4, padx=5, sticky="w")

        ttk.Entry(
            settings,
            textvariable=sigma_var,
            width=10
        ).grid(row=0, column=5, padx=5)

        # ---------------------------------------
        # Figure (created before processing)
        # ---------------------------------------

        fig = Figure(figsize=(11, 4), dpi=90)

        axes = fig.subplots(1, 4)

        for ax in axes:
            ax.axis("off")

        fig.tight_layout()

        # ---------------------------------------
        # Processing function
        # ---------------------------------------

        def run_matching():

            if selected["image"] is None:
                messagebox.showwarning(
                    "Missing Image",
                    "Please choose the original image.",
                    parent=window
                )
                return

            if selected["template"] is None:
                messagebox.showwarning(
                    "Missing Template",
                    "Please choose a template.",
                    parent=window
                )
                return

            try:
                angle = float(angle_var.get())
                threshold = float(threshold_var.get())
                sigma = float(sigma_var.get())

                if not all(
                    np.isfinite(v)
                    for v in (angle, threshold, sigma)
                ):
                    raise ValueError(
                        "Please enter finite numeric values."
                    )

                if not 0 <= threshold <= 1:
                    raise ValueError(
                        "Correlation threshold must be between 0 and 1."
                    )

                if sigma <= 0:
                    raise ValueError(
                        "Canny sigma must be greater than zero."
                    )

                # Original image in grayscale
                image = np.asarray(
                    ImageOps.grayscale(selected["image"]),
                    dtype=np.float64
                )

                # Template in grayscale
                template = np.asarray(
                    ImageOps.grayscale(selected["template"]),
                    dtype=np.float64
                )

                # Rotate template
                rotated_template = transform.rotate(
                    template,
                    angle,
                    resize=True,
                    preserve_range=True
                )

                image_h, image_w = image.shape
                temp_h, temp_w = rotated_template.shape

                if (
                    temp_h > image_h
                    or temp_w > image_w
                ):
                    raise ValueError(
                        "The rotated template is larger than "
                        "the original image."
                    )

                if np.std(rotated_template) < 1e-10:
                    raise ValueError(
                        "The template has no intensity variation."
                    )

                # Normalized cross-correlation
                corr = feature.match_template(
                    image,
                    rotated_template,
                    pad_input=True
                )

                # Valid template centers
                valid = np.zeros(
                    corr.shape,
                    dtype=bool
                )

                start_y = temp_h // 2
                start_x = temp_w // 2

                valid[
                    start_y:start_y + image_h - temp_h + 1,
                    start_x:start_x + image_w - temp_w + 1
                ] = True

                valid_scores = np.where(
                    valid,
                    corr,
                    -np.inf
                )

                best_y, best_x = np.unravel_index(
                    np.argmax(valid_scores),
                    valid_scores.shape
                )

                best_score = float(
                    valid_scores[best_y, best_x]
                )

                # Detect distinct matching peaks
                peaks = feature.peak_local_max(
                    corr,
                    min_distance=max(
                        1,
                        min(temp_h, temp_w) // 4
                    ),
                    threshold_abs=threshold,
                    exclude_border=False
                )

                points = [
                    (int(y), int(x))
                    for y, x in peaks
                    if valid[y, x]
                ]

                # Canny edge detection
                normalized_image = image / 255.0

                edges = feature.canny(
                    normalized_image,
                    sigma=sigma
                )

                # Superimposition
                overlay = np.asarray(
                    selected["image"].convert("RGB")
                ).copy()

                # Green edges
                overlay[edges] = [0, 255, 0]

                overlay_image = Image.fromarray(overlay)

                draw = ImageDraw.Draw(overlay_image)

                # Red circles for matching positions
                for y, x in points:

                    radius = 6

                    draw.ellipse(
                        (
                            x - radius,
                            y - radius,
                            x + radius,
                            y + radius
                        ),
                        outline=(255, 0, 0),
                        width=2
                    )

                # ---------------------------------------
                # Display four results
                # ---------------------------------------

                for ax in axes:
                    ax.clear()

                axes[0].imshow(
                    image,
                    cmap="gray"
                )
                axes[0].set_title("Original Image")

                axes[1].imshow(
                    rotated_template,
                    cmap="gray"
                )
                axes[1].set_title(
                    f"Rotated Template ({angle}°)"
                )

                axes[2].imshow(
                    corr,
                    cmap="gray",
                    vmin=-1,
                    vmax=1
                )
                axes[2].set_title(
                    "Cross-Correlation"
                )

                axes[3].imshow(
                    overlay_image
                )
                axes[3].set_title(
                    "Canny + Matching Overlay"
                )

                for ax in axes:
                    ax.axis("off")

                fig.tight_layout()
                canvas.draw()

                # Save output
                result["image"] = overlay_image
                result["score"] = best_score
                result["angle"] = angle
                result["threshold"] = threshold
                result["matches"] = len(points)

                status_var.set(
                    f"Best correlation: {best_score:.4f}    "
                    f"Matches: {len(points)}    "
                    f"Angle: {angle}°"
                )

            except ValueError as error:
                messagebox.showerror(
                    "Invalid Input",
                    str(error),
                    parent=window
                )

            except Exception as error:
                messagebox.showerror(
                    "Processing Error",
                    str(error),
                    parent=window
                )

        # ---------------------------------------
        # BUTTONS AT THE TOP
        # ---------------------------------------

        button_frame = ttk.Frame(controls)
        button_frame.pack(fill="x", pady=10)

        ttk.Button(
            button_frame,
            text="Run Template Matching",
            command=run_matching
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame,
            text="Done",
            command=window.destroy
        ).pack(side="left", padx=5)

        ttk.Label(
            controls,
            textvariable=status_var
        ).pack(anchor="w", pady=5)

        # ---------------------------------------
        # Graphs below buttons
        # ---------------------------------------

        plot_frame = ttk.Frame(window)
        plot_frame.pack(
            side="top",
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        canvas = FigureCanvasTkAgg(
            fig,
            master=plot_frame
        )

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        canvas.draw()

        # ---------------------------------------
        # Finish
        # ---------------------------------------

        window.protocol(
            "WM_DELETE_WINDOW",
            window.destroy
        )

        window.grab_set()
        parent.wait_window(window)

        if result["image"] is None:
            return None

        return ToolResult(
            image=result["image"],
            message="Template matching completed.",
            details={
                "Operation": "Template matching with Canny overlay",
                "Rotation angle": result["angle"],
                "Correlation threshold": result["threshold"],
                "Maximum correlation": result["score"],
                "Detected matches": result["matches"]
            }
        )
