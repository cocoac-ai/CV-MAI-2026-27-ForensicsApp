
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy import ndimage
from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult


class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Convolution"
    category = "Set 3 · Filtering & Edges"
    description = "Apply a custom kernel."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:

        # Create popup
        window = tk.Toplevel(parent)
        window.title("Custom Convolution")
        window.geometry("850x650")

        ttk.Label(window, text="Number of rows:").pack()
        rows_entry = ttk.Entry(window, width=10)
        rows_entry.pack()
        rows_entry.insert(0, "3")

        ttk.Label(window, text="Number of columns:").pack()
        cols_entry = ttk.Entry(window, width=10)
        cols_entry.pack()
        cols_entry.insert(0, "3")

        # Kernel boxes
        grid = ttk.Frame(window)
        grid.pack(pady=10)

        boxes = []

        def create_matrix():
            try:
                rows = int(rows_entry.get())
                cols = int(cols_entry.get())

                if rows < 1 or cols < 1 or rows > 15 or cols > 15:
                    raise ValueError()

            except ValueError:
                messagebox.showerror(
                    "Invalid size",
                    "Enter a size between 1 and 15.",
                    parent=window
                )
                return

            # Remove old boxes
            for widget in grid.winfo_children():
                widget.destroy()

            boxes.clear()

            # Create new boxes
            for row in range(rows):
                line = []

                for col in range(cols):
                    box = ttk.Entry(grid, width=5, justify="center")
                    box.grid(row=row, column=col, padx=2, pady=2)
                    box.insert(0, "0")
                    line.append(box)

                boxes.append(line)

        ttk.Button(
            window,
            text="Create Matrix",
            command=create_matrix
        ).pack(pady=5)

        # Divisor
        ttk.Label(window, text="Divide by:").pack()
        divisor_entry = ttk.Entry(window, width=10)
        divisor_entry.pack()
        divisor_entry.insert(0, "1")

        def read_kernel():
            rows = []

            for line in boxes:
                numbers = []

                for box in line:
                    numbers.append(float(box.get()))

                rows.append(numbers)

            return np.array(rows)

        result = None

        def apply():
            nonlocal result

            try:
                # Read the kernel
                kernel = read_kernel()

                # Read and apply divisor
                divide_by = float(divisor_entry.get())

                if divide_by == 0:
                    raise ValueError()

                kernel = kernel / divide_by

                if not np.isfinite(kernel).all():
                    raise ValueError()

                # Get current image as grayscale
                image = np.array(document.current.convert("L"))

                # Apply convolution
                filtered = ndimage.convolve(image.astype(float), kernel)

                # Convert result back to an image
                filtered = np.clip(filtered, 0, 255)
                result = Image.fromarray(filtered.astype("uint8"))

                window.destroy()

            except (ValueError, TypeError, RuntimeError):
                messagebox.showerror(
                    "Invalid input",
                    "Enter valid kernel values and a nonzero divisor.",
                    parent=window
                )

        # Apply and Cancel buttons
        buttons = ttk.Frame(window)
        buttons.pack(pady=15)

        ttk.Button(
            buttons,
            text="Apply",
            command=apply
        ).pack(side="left", padx=10)

        ttk.Button(
            buttons,
            text="Cancel",
            command=window.destroy
        ).pack(side="left", padx=10)

        # Show a 3x3 matrix when the popup opens
        create_matrix()

        window.grab_set()
        parent.wait_window(window)

        if result is None:
            return None

        return ToolResult(
            image=result,
            message="Convolution applied."
        )
