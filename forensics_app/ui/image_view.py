"""Canvas that displays a PIL image scaled to the available space."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


class ImageView(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=8)
        self.canvas = tk.Canvas(
            self,
            background="#20242b",
            borderwidth=0,
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self._source: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._resize_job: str | None = None
        self.show(None)

    def show(self, image: Image.Image | None) -> None:
        self._source = image
        self._render()

    def _on_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(50, self._render)

    def _render(self) -> None:
        self._resize_job = None
        self.canvas.delete("all")
        if self._source is None:
            self.canvas.create_text(
                max(self.canvas.winfo_width() // 2, 1),
                max(self.canvas.winfo_height() // 2, 1),
                text="Open an image to begin",
                fill="#c7ccd4",
                font=("TkDefaultFont", 16),
            )
            self._photo = None
            return

        available_w = max(self.canvas.winfo_width() - 24, 1)
        available_h = max(self.canvas.winfo_height() - 24, 1)

        # Scale so the whole image fits the canvas; small images are enlarged too
        width, height = self._source.size
        scale = min(available_w / width, available_h / height)
        size = (max(round(width * scale), 1), max(round(height * scale), 1))

        # Enlarging: NEAREST keeps pixels sharp (thin HOG/edge lines stay crisp).
        # Shrinking: LANCZOS averages pixels, so thin lines do not just vanish.
        method = Image.Resampling.NEAREST if scale > 1 else Image.Resampling.LANCZOS
        preview = self._source.resize(size, method)
        self._photo = ImageTk.PhotoImage(preview)
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self._photo,
            anchor="center",
        )
