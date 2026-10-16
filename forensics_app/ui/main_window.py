"""The application's main window and event coordination."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import UnidentifiedImageError

from forensics_app.core import ImageDocument
from forensics_app.tools.base import ForensicsTool
from forensics_app.tools.registry import ToolRegistry
from .image_view import ImageView


OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
SAVE_TYPES = [("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("TIFF image", "*.tiff")]


class MainWindow:
    def __init__(self, root: tk.Tk, registry: ToolRegistry) -> None:
        self.root = root
        self.registry = registry
        self.document = ImageDocument()
        self.status = tk.StringVar(value="Ready. Open an image to begin.")

        self._configure_window()
        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()
        self._refresh()

    def _configure_window(self) -> None:
        self.root.title("ForensicsApp")
        self.root.geometry("1180x720")
        self.root.minsize(820, 520)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Sidebar.TFrame", background="#eef1f5")
        style.configure("Category.TLabel", background="#eef1f5", font=("TkDefaultFont", 10, "bold"))
        style.configure("Tool.TButton", anchor="w", padding=(10, 7))
        style.configure("Title.TLabel", font=("TkDefaultFont", 16, "bold"))

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open image…", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save result as…", command=self.save_image, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menu.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menu, tearoff=False)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_command(label="Reset to original", command=self.reset)
        menu.add_cascade(label="Edit", menu=edit_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=menu)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        toolbar = ttk.Frame(container, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Open image", command=self.open_image).pack(side="left")
        ttk.Button(toolbar, text="Save result", command=self.save_image).pack(side="left", padx=(6, 0))
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        self.undo_button = ttk.Button(toolbar, text="Undo", command=self.undo)
        self.undo_button.pack(side="left")
        self.redo_button = ttk.Button(toolbar, text="Redo", command=self.redo)
        self.redo_button.pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side="left", padx=(6, 0))

        body = ttk.Panedwindow(container, orient="horizontal")
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, style="Sidebar.TFrame", width=235)
        sidebar.pack_propagate(False)
        body.add(sidebar, weight=0)
        tools_frame = self._build_scrollable_sidebar(sidebar)

        ttk.Label(tools_frame, text="Forensics tools", style="Title.TLabel", background="#eef1f5").pack(
            anchor="w", pady=(0, 12)
        )
        for category, tools in self.registry.categories():
            ttk.Label(tools_frame, text=category, style="Category.TLabel").pack(anchor="w", pady=(9, 4))
            for tool in tools:
                button = ttk.Button(
                    tools_frame,
                    text=tool.title,
                    style="Tool.TButton",
                    command=lambda selected=tool: self.run_tool(selected),
                )
                button.pack(fill="x", pady=2)
                button.bind("<Enter>", lambda _event, selected=tool: self.status.set(selected.description))
                button.bind("<Leave>", lambda _event: self.status.set("Ready."))

        self.image_view = ImageView(body)
        body.add(self.image_view, weight=1)

        inspector = ttk.Frame(body, padding=12, width=250)
        inspector.pack_propagate(False)
        body.add(inspector, weight=0)
        ttk.Label(inspector, text="Results", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        self.results = ttk.Treeview(inspector, columns=("value",), show="tree headings", height=15)
        self.results.heading("#0", text="Property")
        self.results.heading("value", text="Value")
        self.results.column("#0", width=95, stretch=True)
        self.results.column("value", width=120, stretch=True)
        self.results.pack(fill="both", expand=True)

        ttk.Label(container, textvariable=self.status, anchor="w", padding=(10, 6), relief="sunken").pack(fill="x")

    def _build_scrollable_sidebar(self, sidebar: ttk.Frame) -> ttk.Frame:
        """Put a scrollable frame inside the sidebar and return it.

        A Frame cannot scroll by itself, so the tool buttons go into a Frame
        that is placed inside a Canvas; the Scrollbar moves the Canvas view.
        """
        canvas = tk.Canvas(sidebar, background="#eef1f5", highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        tools_frame = ttk.Frame(canvas, style="Sidebar.TFrame", padding=12)
        frame_id = canvas.create_window((0, 0), window=tools_frame, anchor="nw")

        # When the buttons change size, update how far the canvas can scroll
        tools_frame.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        # When the sidebar is resized, make the buttons as wide as the canvas
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(frame_id, width=event.width))

        def on_mousewheel(event: tk.Event) -> None:
            # Only scroll when the mouse is over the sidebar
            widget = self.root.winfo_containing(event.x_root, event.y_root)
            if widget is None or not str(widget).startswith(str(canvas)):
                return
            # Nothing to scroll when all tools already fit
            if canvas.yview() == (0.0, 1.0):
                return
            if event.num == 4 or event.delta > 0:  # Linux uses buttons 4/5
                canvas.yview_scroll(-1, "units")
            else:
                canvas.yview_scroll(1, "units")

        self.root.bind_all("<MouseWheel>", on_mousewheel)  # Windows and macOS
        self.root.bind_all("<Button-4>", on_mousewheel)    # Linux scroll up
        self.root.bind_all("<Button-5>", on_mousewheel)    # Linux scroll down
        return tools_frame

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<Control-o>", lambda _event: self.open_image())
        self.root.bind_all("<Control-Shift-S>", lambda _event: self.save_image())
        self.root.bind_all("<Control-z>", lambda _event: self.undo())
        self.root.bind_all("<Control-y>", lambda _event: self.redo())

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(title="Open evidence image", filetypes=OPEN_TYPES)
        if not filename:
            return
        try:
            self.document.load(filename)
        except (OSError, UnidentifiedImageError) as error:
            messagebox.showerror("Could not open image", str(error), parent=self.root)
            return
        self.status.set(f"Opened {Path(filename).name}")
        self._show_default_details()
        self._refresh()

    def save_image(self) -> None:
        if not self._require_image():
            return
        source = self.document.path
        initial = f"{source.stem}_result.png" if source else "result.png"
        filename = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            initialfile=initial,
            filetypes=SAVE_TYPES,
        )
        if not filename:
            return
        try:
            self.document.save(filename)
        except OSError as error:
            messagebox.showerror("Could not save image", str(error), parent=self.root)
            return
        self.status.set(f"Saved result as {Path(filename).name}")

    def run_tool(self, tool: ForensicsTool) -> None:
        if tool.requires_image and not self._require_image():
            return
        try:
            result = tool.run(self.root, self.document)
        except Exception as error:  # keep one student feature from crashing the shell
            messagebox.showerror(f"{tool.title} failed", str(error), parent=self.root)
            self.status.set(f"Error in {tool.title}.")
            return
        if result is None:
            self.status.set(f"Cancelled {tool.title}.")
            return
        if result.image is not None:
            self.document.apply(result.image)
        self._show_details(result.details)
        self.status.set(result.message)
        self._refresh()

    def undo(self) -> None:
        if self.document.undo():
            self.status.set("Undid the last image operation.")
            self._show_default_details()
            self._refresh()

    def redo(self) -> None:
        if self.document.redo():
            self.status.set("Redid the image operation.")
            self._show_default_details()
            self._refresh()

    def reset(self) -> None:
        if self.document.reset():
            self.status.set("Restored the original image.")
            self._show_default_details()
            self._refresh()

    def _require_image(self) -> bool:
        if self.document.is_loaded:
            return True
        messagebox.showinfo("No image loaded", "Open an image first.", parent=self.root)
        return False

    def _refresh(self) -> None:
        self.image_view.show(self.document.current)
        self.undo_button.configure(state="normal" if self.document.can_undo else "disabled")
        self.redo_button.configure(state="normal" if self.document.can_redo else "disabled")
        title = self.document.path.name if self.document.path else "No image"
        self.root.title(f"ForensicsApp — {title}")

    def _show_default_details(self) -> None:
        image = self.document.current
        if image is None:
            self._show_details({})
            return
        self._show_details(
            {
                "File": self.document.path.name if self.document.path else "—",
                "Size": f"{image.width} × {image.height}",
                "Mode": image.mode,
            }
        )

    def _show_details(self, details: dict[str, object]) -> None:
        for item in self.results.get_children():
            self.results.delete(item)
        for name, value in details.items():
            self.results.insert("", "end", text=str(name), values=(str(value),))

    def show_about(self) -> None:
        messagebox.showinfo(
            "About ForensicsApp",
            "A modular image-forensics application for the Computer Vision course.\n\n"
            "Add each weekly feature as a tool in forensics_app/tools/.",
            parent=self.root,
        )
