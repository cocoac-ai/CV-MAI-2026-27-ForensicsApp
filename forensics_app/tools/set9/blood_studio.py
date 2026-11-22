"""Set 9.3 Blood Cell Investigation Studio.

Runs inference in the Mac-safe subprocess provided by yolo_detection.py.
Educational/research demonstration; no patient-level or clinical conclusions.
"""
from __future__ import annotations

import base64
import csv
import html
import io
from collections import Counter
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageTk

from forensics_app.core import ImageDocument
from ..base import ToolResult
from .yolo_detection import _model_names, _predict_safely, _image_quality, _label_font, COLORS


METHODS = ("Original", "Higher contrast", "Sharpened", "Grayscale")
NOTICE = ("Educational image-level counts only. Not a clinical blood count, "
          "medical diagnosis, or identification of a biological sample.")


def _prepare(image: Image.Image, method: str) -> Image.Image:
    image = image.convert("RGB")
    if method == "Higher contrast":
        return ImageEnhance.Contrast(image).enhance(1.5)
    if method == "Sharpened":
        return image.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
    if method == "Grayscale":
        return image.convert("L").convert("RGB")
    return image.copy()


def _png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class BloodStudio:
    def __init__(self, parent: tk.Misc, image: Image.Image, source: Path):
        self.image = image.convert("RGB").copy()
        self.source = source
        self.names = _model_names(str(source))
        self.data: dict[str, list[dict]] = {}
        self.manual_log: list[str] = []
        self.last_output: Image.Image | None = None
        self.method = tk.StringVar(value="Original")
        self.conf = tk.DoubleVar(value=0.25)
        self.iou = tk.DoubleVar(value=0.45)
        self.scores = tk.BooleanVar(value=False)
        self.only_uncertain = tk.BooleanVar(value=False)
        self.class_label = tk.StringVar(value=next(iter(self.names.values())))
        self.info = tk.StringVar(value="Run detection to begin")
        self.statistics = tk.StringVar(value="No detections yet")
        self.compare_info = tk.StringVar(value="Compare methods using the same original image.")
        self.quality = tk.StringVar(value="")
        self.result: ToolResult | None = None
        self.photo = None
        self.crop_photo = None
        self.preview_size = (700, 490)
        self.render_ratio = 1.0
        self.render_origin = (0, 0)
        self.window = tk.Toplevel(parent)
        self.window.title("Set 9.3 — Blood Cell Investigation Studio")
        self.window.geometry("1180x820")
        self.window.minsize(970, 680)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._finish)
        self._build()
        self._show_image(self.image)
        self.window.grab_set()
        self.window.wait_window()

    def _build(self):
        header = ttk.Frame(self.window, padding=(15, 10))
        header.pack(fill="x")
        ttk.Label(header, text="BLOOD CELL INVESTIGATION STUDIO",
                  font=("Arial", 17, "bold")).pack(side="left")
        ttk.Label(header, text="YOLO11 · Set 9.3", foreground="#586576").pack(side="right")
        self.tabs = ttk.Notebook(self.window)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        self.analysis_tab = ttk.Frame(self.tabs, padding=8)
        self.compare_tab = ttk.Frame(self.tabs, padding=12)
        self.report_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.analysis_tab, text="1 · Detect & review")
        self.tabs.add(self.compare_tab, text="2 · Preprocessing comparison")
        self.tabs.add(self.report_tab, text="3 · Export report")
        self._build_analysis()
        self._build_comparison()
        self._build_report()
        bottom = ttk.Frame(self.window, padding=(12, 5))
        bottom.pack(fill="x")
        ttk.Label(bottom, text=NOTICE, foreground="#6d747f", wraplength=850).pack(side="left", fill="x", expand=True)
        ttk.Button(bottom, text="Close & show result in ForensicsApp", command=self._finish).pack(side="right", padx=(8, 0))

    def _build_analysis(self):
        bar = ttk.Frame(self.analysis_tab)
        bar.pack(fill="x", pady=(0, 8))
        ttk.Label(bar, text="Method:").pack(side="left")
        ttk.Combobox(bar, textvariable=self.method, values=METHODS, state="readonly", width=18).pack(side="left", padx=(5, 15))
        ttk.Label(bar, text="Confidence:").pack(side="left")
        ttk.Scale(bar, from_=0.10, to=0.90, variable=self.conf, command=lambda _v: self._refresh()).pack(side="left", fill="x", expand=True, padx=5)
        self.conf_text = ttk.Label(bar, text="0.25", width=5)
        self.conf_text.pack(side="left")
        ttk.Button(bar, text="Run detection", command=self._run).pack(side="left", padx=(12, 0))
        ttk.Label(self.analysis_tab, textvariable=self.info, foreground="#486581").pack(anchor="w", pady=(0, 7))
        area = ttk.Panedwindow(self.analysis_tab, orient="horizontal")
        area.pack(fill="both", expand=True)
        left = ttk.Frame(area)
        right = ttk.Frame(area)
        area.add(left, weight=3)
        area.add(right, weight=2)
        self.canvas = tk.Canvas(left, bg="#202632", highlightthickness=0, width=680, height=490)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._click_canvas)
        self.canvas.bind("<Configure>", lambda _event: self._refresh())
        ttk.Label(left, text="Click any outlined cell or select a row to inspect a crop.",
                  foreground="#5b6570").pack(anchor="w", pady=5)
        ttk.Label(right, textvariable=self.statistics, font=("Arial", 11, "bold"), wraplength=390).pack(fill="x", pady=(0, 8))
        filterrow = ttk.Frame(right)
        filterrow.pack(fill="x")
        ttk.Checkbutton(filterrow, text="Only low-confidence candidates", variable=self.only_uncertain,
                        command=self._refresh).pack(side="left")
        ttk.Checkbutton(filterrow, text="Score labels", variable=self.scores,
                        command=self._refresh).pack(side="right")
        listing = ttk.Frame(right)
        listing.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(listing, columns=("type", "confidence", "status"), show="headings", height=9)
        for col, label, width in (("type", "Class", 95), ("confidence", "Confidence", 88), ("status", "Review", 85)):
            self.tree.heading(col, text=label)
            self.tree.column(col, width=width, anchor="center")
        scroll = ttk.Scrollbar(listing, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._select_cell)
        self.crop = ttk.Label(right, text="Select a cell to zoom", anchor="center")
        self.crop.pack(fill="x", pady=8)
        row = ttk.Frame(right)
        row.pack(fill="x")
        ttk.Button(row, text="Accept", command=lambda: self._decision("accepted")).pack(side="left", padx=(0, 4))
        ttk.Button(row, text="Reject", command=lambda: self._decision("rejected")).pack(side="left", padx=4)
        ttk.Button(row, text="Restore", command=lambda: self._decision("normal")).pack(side="left", padx=4)
        row2 = ttk.Frame(right)
        row2.pack(fill="x", pady=6)
        ttk.Label(row2, text="Correct class:").pack(side="left")
        ttk.Combobox(row2, textvariable=self.class_label, values=list(self.names.values()),
                     state="readonly", width=13).pack(side="left", padx=5)
        ttk.Button(row2, text="Apply", command=self._correct_class).pack(side="left")
        ttk.Label(right, text="Review decisions update the image, counts, and exported report.",
                  wraplength=375, foreground="#596575").pack(anchor="w", pady=5)

    def _build_comparison(self):
        ttk.Label(self.compare_tab, text="Compare how preprocessing changes YOLO's detections",
                  font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Label(self.compare_tab, text="Each method is actually passed through the same trained model. "
                  "Counts may change; more boxes do not necessarily mean better accuracy.",
                  wraplength=850).pack(anchor="w")
        ttk.Button(self.compare_tab, text="Run all 4 methods", command=self._run_all).pack(anchor="w", pady=12)
        self.compare_table = ttk.Treeview(self.compare_tab, columns=("method", "total", "pc", "rbc", "wbc", "quality"),
                                          show="headings", height=6)
        for col, label, width in (("method", "Preprocessing", 170), ("total", "Total", 75),
                                  ("pc", "PC", 70), ("rbc", "RBC", 70), ("wbc", "WBC", 70),
                                  ("quality", "Image quality warning", 250)):
            self.compare_table.heading(col, text=label)
            self.compare_table.column(col, width=width, anchor="center")
        self.compare_table.pack(fill="x", pady=6)
        ttk.Label(self.compare_tab, textvariable=self.compare_info, wraplength=900).pack(anchor="w", pady=8)
        ttk.Label(self.compare_tab, text="Select any method on the first tab and run it to examine individual cells.",
                  foreground="#626e79").pack(anchor="w")

    def _build_report(self):
        ttk.Label(self.report_tab, text="Export an auditable image-analysis report",
                  font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 12))
        txt = ("The HTML report includes the original and annotated images, model and settings, "
               "cell counts, individual predictions, confidence scores, manual corrections, "
               "a quality warning, and any preprocessing comparison results. Open the HTML "
               "in your browser to view or print it. CSV is also available.")
        ttk.Label(self.report_tab, text=txt, wraplength=900).pack(anchor="w", pady=(0, 20))
        ttk.Button(self.report_tab, text="Export HTML report (with images)",
                   command=self._export_html).pack(anchor="w", pady=7)
        ttk.Button(self.report_tab, text="Export detection CSV", command=self._export_csv).pack(anchor="w", pady=7)
        ttk.Button(self.report_tab, text="Save annotated PNG", command=self._export_image).pack(anchor="w", pady=7)
        ttk.Label(self.report_tab, text=NOTICE, foreground="#a05b31", wraplength=800).pack(anchor="w", pady=20)

    def _entries(self):
        return self.data.get(self.method.get(), [])

    def _visible(self):
        threshold = self.conf.get()
        return [d for d in self._entries() if d["status"] != "rejected" and
                (d["status"] == "accepted" or d["confidence"] >= threshold)]

    def _run(self):
        method = self.method.get()
        self.info.set(f"Running {method} detection on CPU…")
        self.window.update_idletasks()
        try:
            raw = _predict_safely(_prepare(self.image, method), str(self.source),
                                  0.10, self.iou.get())
            self.data[method] = [{"id": index, "class_id": int(r["class_id"]),
                                  "name": self.names[int(r["class_id"])],
                                  "confidence": float(r["confidence"]),
                                  "xyxy": tuple(r["xyxy"]), "status": "normal"}
                                 for index, r in enumerate(raw, start=1)]
            self.info.set(f"{method}: {len(raw)} candidates at 0.10 inference threshold · adjust display threshold above")
            self._refresh()
        except Exception as exc:
            self.info.set("Detection failed")
            messagebox.showerror("YOLO detection", str(exc), parent=self.window)

    def _run_all(self):
        for method in METHODS:
            self.method.set(method)
            self._run()
            if method not in self.data:
                break
        self._comparison_update()
        self.tabs.select(self.compare_tab)

    def _comparison_update(self):
        for iid in self.compare_table.get_children():
            self.compare_table.delete(iid)
        for method in METHODS:
            if method not in self.data:
                continue
            visible = [d for d in self.data[method] if d["status"] != "rejected" and
                       (d["status"] == "accepted" or d["confidence"] >= self.conf.get())]
            counts = Counter(d["name"] for d in visible)
            warning, _, _ = _image_quality(_prepare(self.image, method))
            def tally(key):
                return sum(n for label, n in counts.items() if label.upper() == key)
            self.compare_table.insert("", "end", values=(method, len(visible), tally("PC"),
                                                    tally("RBC"), tally("WBC"), warning))
        self.compare_info.set(f"Comparison uses confidence ≥ {self.conf.get():.2f}, "
                              "except manually accepted cells. Changes in counts are not proof of improved accuracy.")

    def _render(self) -> Image.Image:
        method = self.method.get()
        processed = _prepare(self.image, method)
        out = processed.copy()
        drawer = ImageDraw.Draw(out)
        font = _label_font(out.width, out.height)
        for d in self._visible():
            box = d["xyxy"]
            color = COLORS[d["class_id"] % len(COLORS)]
            xy = tuple(int(v) for v in box)
            drawer.rectangle(xy, outline=color, width=max(2, min(out.size)//250))
            if self.scores.get():
                label = f"{d['name']} {d['confidence']*100:.0f}%"
                drawer.text((max(0, xy[0]), max(0, xy[1]-18)), label, fill=color, font=font,
                            stroke_width=1, stroke_fill="black")
        return out

    def _show_image(self, image: Image.Image):
        self.canvas.delete("all")
        width = max(200, self.canvas.winfo_width())
        height = max(200, self.canvas.winfo_height())
        self.render_ratio = min(width / image.width, height / image.height)
        size = (max(1, int(image.width * self.render_ratio)),
                max(1, int(image.height * self.render_ratio)))
        displayed = image.resize(size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(displayed, master=self.window)
        x, y = (width-size[0])//2, (height-size[1])//2
        self.render_origin = x, y
        self.canvas.create_image(x, y, image=self.photo, anchor="nw")

    def _refresh(self):
        if not hasattr(self, "canvas"):
            return
        self.conf_text.configure(text=f"{self.conf.get():.2f}")
        if self.method.get() not in self.data:
            self._show_image(_prepare(self.image, self.method.get()))
            return
        self._show_image(self._render())
        saved = self.tree.selection()
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        visible = self._visible()
        for d in self._entries():
            if self.only_uncertain.get() and d["confidence"] >= self.conf.get() and d["status"] != "rejected":
                continue
            self.tree.insert("", "end", iid=str(d["id"]), values=(d["name"],
                       f"{d['confidence']*100:.1f}%", d["status"]))
        if saved and self.tree.exists(saved[0]):
            self.tree.selection_set(saved[0])
        counts = Counter(d["name"] for d in visible)
        summary = "   ·   ".join(f"{name}: {counts.get(name,0)}" for name in self.names.values())
        self.statistics.set(f"{len(visible)} shown  ·  {summary}")
        self.last_output = self._render()
        self._comparison_update()

    def _find_selected(self):
        current = self.tree.selection()
        if not current:
            return None
        return next((d for d in self._entries() if d["id"] == int(current[0])), None)

    def _select_cell(self, _event=None):
        selected = self._find_selected()
        if selected is None:
            return
        self.class_label.set(selected["name"])
        x1,y1,x2,y2 = map(int, selected["xyxy"])
        pad = 18
        x1,y1 = max(0,x1-pad),max(0,y1-pad)
        x2,y2 = min(self.image.width,x2+pad),min(self.image.height,y2+pad)
        if x2 <= x1 or y2 <= y1:
            return
        cropped = _prepare(self.image, self.method.get()).crop((x1,y1,x2,y2))
        cropped.thumbnail((300,165), Image.Resampling.LANCZOS)
        self.crop_photo = ImageTk.PhotoImage(cropped, master=self.window)
        self.crop.configure(image=self.crop_photo, text="")

    def _click_canvas(self, event):
        if not self._entries():
            return
        x = (event.x-self.render_origin[0]) / self.render_ratio
        y = (event.y-self.render_origin[1]) / self.render_ratio
        hits = [d for d in self._entries() if d["xyxy"][0] <= x <= d["xyxy"][2] and
                d["xyxy"][1] <= y <= d["xyxy"][3]]
        if hits:
            # Most specific overlapping box wins.
            hit = min(hits, key=lambda d: (d["xyxy"][2]-d["xyxy"][0])*(d["xyxy"][3]-d["xyxy"][1]))
            iid = str(hit["id"])
            if self.tree.exists(iid):
                self.tree.selection_set(iid)
                self.tree.see(iid)
                self._select_cell()

    def _decision(self, action: str):
        d = self._find_selected()
        if d is None:
            messagebox.showinfo("Select a cell", "Select a row or click a cell first.", parent=self.window)
            return
        before = d["status"]
        d["status"] = action
        self.manual_log.append(f"{self.method.get()} cell #{d['id']}: {before} → {action}")
        self._refresh()

    def _correct_class(self):
        d = self._find_selected()
        if d is None:
            return
        new_name = self.class_label.get()
        new_id = next((idx for idx, name in self.names.items() if name == new_name), None)
        if new_id is None:
            return
        old_name = d["name"]
        d["name"] = new_name
        d["class_id"] = new_id
        d["status"] = "accepted"
        self.manual_log.append(f"{self.method.get()} cell #{d['id']}: {old_name} → {new_name} (manual correction)")
        self._refresh()

    def _report_rows(self):
        return self._entries()

    def _ensure_results(self):
        if self.method.get() not in self.data:
            messagebox.showwarning("No detections", "Run detection before exporting.", parent=self.window)
            return False
        return True

    def _export_image(self):
        if not self._ensure_results():
            return
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".png",
                    filetypes=[("PNG image", "*.png")], title="Save annotated image")
        if path:
            self._render().save(path)

    def _export_csv(self):
        if not self._ensure_results():
            return
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
                    filetypes=[("CSV file", "*.csv")], title="Export detections")
        if not path:
            return
        with open(path, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["method", "id", "class", "confidence", "x1", "y1", "x2", "y2", "review_status", "displayed"])
            visible_ids = {d["id"] for d in self._visible()}
            for d in self._report_rows():
                writer.writerow([self.method.get(), d["id"], d["name"], f"{d['confidence']:.4f}",
                                 *[round(v,2) for v in d["xyxy"]], d["status"], d["id"] in visible_ids])
        messagebox.showinfo("Exported", f"Saved {path}", parent=self.window)

    def _export_html(self):
        if not self._ensure_results():
            return
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".html",
                     filetypes=[("HTML report", "*.html")], title="Export full report")
        if not path:
            return
        method = self.method.get()
        visible = self._visible()
        counts = Counter(d["name"] for d in visible)
        warning, sharpness, brightness = _image_quality(_prepare(self.image, method))
        image_tag = _png_data_uri(self._render())
        original_tag = _png_data_uri(self.image)
        def esc(value):
            return html.escape(str(value), quote=True)
        countrows = "".join(f"<tr><td>{esc(n)}</td><td>{counts.get(n,0)}</td></tr>" for n in self.names.values())
        detrows = "".join("<tr>"+"".join(f"<td>{esc(x)}</td>" for x in (
                   d["id"], d["name"], f"{d['confidence']:.3f}",
                   str(tuple(round(v,1) for v in d["xyxy"])), d["status"]))+"</tr>" for d in self._entries())
        comparerows = "".join("<tr>"+"".join(f"<td>{esc(v)}</td>" for v in self.compare_table.item(iid,"values"))+"</tr>"
                              for iid in self.compare_table.get_children())
        reviews = "".join(f"<li>{esc(v)}</li>" for v in self.manual_log) or "<li>No manual review changes</li>"
        content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Blood cell analysis report</title>
<style>body{{font-family:Arial,sans-serif;max-width:1050px;margin:35px auto;color:#253246;line-height:1.5}}
h1{{color:#154b70}}.cols{{display:flex;gap:18px}}.cols>div{{flex:1}}img{{max-width:100%;border:1px solid #ddd}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}td,th{{border:1px solid #ddd;padding:7px;text-align:left}}
th{{background:#edf3f8}}.note{{background:#fff5e6;padding:12px}}@media print{{body{{margin:10mm}}}}</style>
<h1>Blood Cell Investigation Report</h1><p>Generated: {esc(datetime.now().isoformat(timespec="seconds"))}</p>
<p>Model: {esc(self.source.name)} · Method: {esc(method)} · Confidence threshold: {self.conf.get():.2f} · NMS IoU: {self.iou.get():.2f}</p>
<p class="note">{esc(NOTICE)}</p><h2>Microscope image and detections</h2>
<div class="cols"><div><h3>Original</h3><img src="{original_tag}" alt="Original image"></div>
<div><h3>Reviewed predictions</h3><img src="{image_tag}" alt="Annotated image"></div></div>
<h2>Counts in this image</h2><table><tr><th>Class</th><th>Displayed detections</th></tr>{countrows}</table>
<p>Total displayed: {len(visible)} · Quality warning: {esc(warning)} · Sharpness heuristic: {sharpness:.1f} · Mean brightness: {brightness:.1f}</p>
<h2>All model candidates and review status</h2><table><tr><th>ID</th><th>Class</th><th>Confidence</th><th>Coordinates (pixels)</th><th>Status</th></tr>{detrows}</table>
<h2>Preprocessing comparison</h2><table><tr><th>Method</th><th>Total</th><th>PC</th><th>RBC</th><th>WBC</th><th>Quality note</th></tr>{comparerows}</table>
<h2>Manual review history</h2><ul>{reviews}</ul></html>'''
        Path(path).write_text(content, encoding="utf-8")
        messagebox.showinfo("Report exported", f"Saved {path}", parent=self.window)

    def _finish(self):
        if self.method.get() in self.data:
            visible = self._visible()
            counts = Counter(d["name"] for d in visible)
            self.result = ToolResult(
                image=self._render(),
                message=f"Blood Cell Studio: {len(visible)} objects shown ({self.method.get()}).",
                details={"Model": self.source.name, "Preprocessing": self.method.get(),
                         "Confidence threshold": f"{self.conf.get():.2f}",
                         "Total cells": len(visible),
                         **{f"{label} detected": counts.get(label,0) for label in self.names.values()},
                         "Manual review changes": len(self.manual_log),
                         "Note": "Educational image-level counts; not medical diagnosis"})
        self.window.destroy()


def open_blood_studio(parent: tk.Misc, document: ImageDocument, model_path: Path) -> ToolResult | None:
    if document.current is None:
        messagebox.showwarning("No image", "Open a microscope image first.", parent=parent)
        return None
    if not model_path.is_file():
        messagebox.showerror("Missing model", f"Trained model not found:\n{model_path}", parent=parent)
        return None
    try:
        studio = BloodStudio(parent, document.current, model_path)
        return studio.result
    except Exception as exc:
        messagebox.showerror("Blood Cell Studio", str(exc), parent=parent)
        return None
