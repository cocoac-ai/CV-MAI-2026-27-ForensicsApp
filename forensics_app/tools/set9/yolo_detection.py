"""Set 9 — YOLO object detection for ForensicsApp.

Provides a general pretrained detector and a blood-smear detection/review tool.
Requires: ultralytics, torch, pillow, numpy. YOLO runs in an isolated Python process for macOS stability.
This is an educational demonstrator, not a clinical diagnostic tool.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..common import MODELS_FOLDER


BLOOD_MODEL = MODELS_FOLDER / "set9-colab" / "blood_cell_best.pt"
GENERAL_MODEL = MODELS_FOLDER / "yolo11n.pt"
COLORS = [(45, 130, 250), (250, 90, 90), (190, 85, 225), (255, 170, 40)]


# Run all PyTorch / Ultralytics operations outside Tkinter's process.
# A segfault in a native library will then produce an error, rather than
# bringing down the entire ForensicsApp window.
_WORKER = r"""
import json
import sys
from pathlib import Path
from ultralytics import YOLO

mode, source, output = sys.argv[1:4]
model = YOLO(source)

if mode == 'names':
    result = {'names': {str(k): str(v) for k, v in model.names.items()}}
else:
    image_path, confidence, iou = sys.argv[4:7]
    prediction = model.predict(
        source=image_path,
        conf=float(confidence),
        iou=float(iou),
        device='cpu',
        verbose=False,
    )[0]
    boxes = []
    for b in prediction.boxes:
        boxes.append({
            'class_id': int(b.cls.item()),
            'confidence': float(b.conf.item()),
            'xyxy': [float(v) for v in b.xyxy[0].tolist()],
        })
    result = {'boxes': boxes}
Path(output).write_text(json.dumps(result), encoding='utf-8')
"""


def _worker(mode: str, model_source: str, image_path: str = "",
            confidence: float = 0.25, iou: float = 0.45) -> dict:
    with tempfile.TemporaryDirectory(prefix="forensics_yolo_") as folder:
        output_path = os.path.join(folder, "output.json")
        args = [sys.executable, "-c", _WORKER, mode, model_source, output_path]
        if mode == "predict":
            args += [image_path, str(confidence), str(iou)]
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        try:
            completed = subprocess.run(
                args, capture_output=True, text=True, timeout=180, env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("YOLO operation timed out after 180 seconds") from exc
        if completed.returncode != 0:
            error = (completed.stderr or completed.stdout or "Unknown worker error").strip()
            raise RuntimeError(f"YOLO process failed (exit {completed.returncode}):\n{error[-2000:]}")
        if not os.path.isfile(output_path):
            raise RuntimeError("YOLO completed but did not produce results")
        with open(output_path, encoding="utf-8") as f:
            return json.load(f)


@lru_cache(maxsize=2)
def _model_names(source: str) -> dict[int, str]:
    return {int(k): v for k, v in _worker("names", source)["names"].items()}


def _predict_safely(image: Image.Image, source: str,
                    confidence: float, iou: float) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="forensics_yolo_image_") as folder:
        image_path = os.path.join(folder, "input.png")
        image.convert("RGB").save(image_path)
        return _worker("predict", source, image_path, confidence, iou)["boxes"]


def _image_quality(image: Image.Image) -> tuple[str, float, float]:
    # Pixel-level sharpness heuristic, avoiding OpenCV in the Tkinter process.
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    laplacian = (
        gray[1:-1, :-2] + gray[1:-1, 2:] +
        gray[:-2, 1:-1] + gray[2:, 1:-1] -
        4 * gray[1:-1, 1:-1]
    ) if min(gray.shape) >= 3 else np.array([0.0])
    sharpness = float(laplacian.var())
    warnings = []
    # Heuristic thresholds: NOT clinically validated quality criteria.
    if sharpness < 60:
        warnings.append("possibly blurry")
    if brightness < 55:
        warnings.append("possibly too dark")
    elif brightness > 215:
        warnings.append("possibly too bright")
    return (", ".join(warnings) if warnings else "No obvious quality warning",
            sharpness, brightness)


def _settings_dialog(parent: tk.Misc, class_names: dict[int, str], *, blood: bool):
    w = tk.Toplevel(parent)
    w.title("Set 9 — " + ("Blood Cell Analysis" if blood else "Pretrained YOLO"))
    w.transient(parent)
    w.resizable(False, False)
    frame = ttk.Frame(w, padding=16)
    frame.pack(fill="both", expand=True)

    confidence = tk.DoubleVar(value=0.25)
    iou = tk.DoubleVar(value=0.45)
    show_scores = tk.BooleanVar(value=True)
    review = tk.BooleanVar(value=blood)
    export = tk.BooleanVar(value=False)
    selected = {key: tk.BooleanVar(value=True) for key in class_names}
    gt_path = tk.StringVar(value="")

    ttk.Label(frame, text="Confidence threshold").pack(anchor="w")
    conf_label = ttk.Label(frame, text="0.25")
    conf_label.pack(anchor="e")
    ttk.Scale(frame, from_=0.10, to=0.90, variable=confidence,
              command=lambda value: conf_label.configure(text=f"{float(value):.2f}"))\
        .pack(fill="x")

    ttk.Label(frame, text="NMS overlap threshold (IoU)").pack(anchor="w", pady=(10, 0))
    iou_label = ttk.Label(frame, text="0.45")
    iou_label.pack(anchor="e")
    ttk.Scale(frame, from_=0.10, to=0.90, variable=iou,
              command=lambda value: iou_label.configure(text=f"{float(value):.2f}"))\
        .pack(fill="x")

    ttk.Label(frame, text="Show categories:").pack(anchor="w", pady=(12, 2))
    classes_frame = ttk.Frame(frame)
    classes_frame.pack(fill="x")
    # Large pretrained COCO models can have 80 categories; show only classes
    # present on the current model as a scrollable list.
    c = tk.Canvas(classes_frame, height=108, width=325, highlightthickness=0)
    scroll = ttk.Scrollbar(classes_frame, orient="vertical", command=c.yview)
    inside = ttk.Frame(c)
    inside.bind("<Configure>", lambda _e: c.configure(scrollregion=c.bbox("all")))
    c.create_window((0, 0), window=inside, anchor="nw")
    c.configure(yscrollcommand=scroll.set)
    c.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    for idx, name in class_names.items():
        ttk.Checkbutton(inside, text=name, variable=selected[idx]).pack(anchor="w")

    ttk.Checkbutton(frame, text="Show confidence scores on boxes",
                    variable=show_scores).pack(anchor="w", pady=(10, 0))
    if blood:
        ttk.Checkbutton(frame, text="Review low-confidence candidates before displaying",
                        variable=review).pack(anchor="w")
    ttk.Checkbutton(frame, text="Export detection CSV after running",
                    variable=export).pack(anchor="w")

    if blood:
        ttk.Label(frame, text="Optional ground-truth YOLO .txt file:")\
            .pack(anchor="w", pady=(10, 2))
        gt_row = ttk.Frame(frame)
        gt_row.pack(fill="x")
        ttk.Entry(gt_row, textvariable=gt_path, width=35).pack(side="left", fill="x", expand=True)
        ttk.Button(gt_row, text="Browse", command=lambda: gt_path.set(
            filedialog.askopenfilename(parent=w, title="Select YOLO ground-truth labels",
                                       filetypes=[("YOLO label file", "*.txt")]) or gt_path.get()
        )).pack(side="left", padx=(6, 0))

    choice = {"value": None}
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=(14, 0))

    def submit():
        picked = {idx for idx, var in selected.items() if var.get()}
        if not picked:
            messagebox.showwarning("No categories", "Select at least one category.", parent=w)
            return
        choice["value"] = {
            "confidence": float(confidence.get()),
            "iou": float(iou.get()),
            "categories": picked,
            "scores": bool(show_scores.get()),
            "review": bool(review.get()) if blood else False,
            "export": bool(export.get()),
            "gt_path": gt_path.get().strip() if blood else "",
        }
        w.destroy()

    ttk.Button(buttons, text="Run detection", command=submit).pack(side="left")
    ttk.Button(buttons, text="Cancel", command=w.destroy).pack(side="left", padx=8)
    w.bind("<Escape>", lambda _event: w.destroy())
    w.grab_set()
    w.wait_window()
    return choice["value"]


def _review_dialog(parent: tk.Misc, detections: list[dict], names: dict[int, str]):
    """Let the user accept/reject uncertain candidates and correct class labels.

    Candidates below the main confidence threshold are NOT included by default.
    """
    uncertain = [d for d in detections if d["status"] == "review"]
    if not uncertain:
        return
    w = tk.Toplevel(parent)
    w.title("Review uncertain detections")
    w.geometry("650x380")
    w.transient(parent)
    f = ttk.Frame(w, padding=12)
    f.pack(fill="both", expand=True)
    ttk.Label(f, text="These candidates were below your confidence threshold. "
                      "Accept only if you have inspected the image.", wraplength=610)\
        .pack(anchor="w", pady=(0, 8))
    tree = ttk.Treeview(f, columns=("id", "class", "confidence", "status"),
                        show="headings", height=10)
    for col, label, width in (("id", "ID", 60), ("class", "Class", 130),
                              ("confidence", "Confidence", 100), ("status", "Decision", 100)):
        tree.heading(col, text=label)
        tree.column(col, width=width)
    tree.pack(fill="both", expand=True)
    for d in uncertain:
        tree.insert("", "end", iid=str(d["id"]), values=(
            d["id"], names[d["class_id"]], f"{d['confidence']:.2f}", "Excluded"))

    def set_status(value: str):
        for iid in tree.selection():
            det = next(d for d in uncertain if d["id"] == int(iid))
            det["status"] = value
            tree.set(iid, "status", "Accepted" if value == "accepted" else "Excluded")

    controls = ttk.Frame(f)
    controls.pack(fill="x", pady=10)
    ttk.Button(controls, text="Accept selected", command=lambda: set_status("accepted"))\
        .pack(side="left")
    ttk.Button(controls, text="Exclude selected", command=lambda: set_status("excluded"))\
        .pack(side="left", padx=5)
    chosen_class = tk.StringVar(value=next(iter(names.values())))
    ttk.Combobox(controls, textvariable=chosen_class, state="readonly", width=14,
                 values=list(names.values())).pack(side="left", padx=(8, 4))

    def reclassify_selected():
        new_id = next((idx for idx, label in names.items()
                       if label == chosen_class.get()), None)
        if new_id is None:
            return
        for iid in tree.selection():
            det = next(d for d in uncertain if d["id"] == int(iid))
            det["class_id"] = new_id
            det["name"] = names[new_id]
            tree.set(iid, "class", names[new_id])

    ttk.Button(controls, text="Correct class", command=reclassify_selected)\
        .pack(side="left")
    ttk.Button(controls, text="Done", command=w.destroy).pack(side="right")
    w.grab_set()
    w.wait_window()


def _read_ground_truth(label_path: str, width: int, height: int, names: dict[int, str]):
    boxes = []
    if not label_path:
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            idx = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
            if idx not in names:
                continue
            boxes.append({"name": names[idx], "xyxy": (
                (xc - bw / 2) * width, (yc - bh / 2) * height,
                (xc + bw / 2) * width, (yc + bh / 2) * height)})
    return boxes


def _label_font(image_width: int, image_height: int) -> ImageFont.ImageFont:
    """Pick a legible font at the image's native resolution, with safe fallbacks."""
    font_size = max(16, min(32, round(min(image_width, image_height) * 0.027)))
    font_paths = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "DejaVuSans-Bold.ttf",  # Pillow-supported fallback
    )
    for path in font_paths:
        try:
            return ImageFont.truetype(path, font_size)
        except (OSError, ValueError):
            continue
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def _draw_box_label(draw: ImageDraw.ImageDraw, image: Image.Image,
                    box: tuple[float, float, float, float], label: str,
                    color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    """Draw a box with an opaque readable tag that stays inside the image."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(image.width - 1, round(x1)))
    x2 = max(0, min(image.width - 1, round(x2)))
    y1 = max(0, min(image.height - 1, round(y1)))
    y2 = max(0, min(image.height - 1, round(y2)))
    if x2 < x1 or y2 < y1:
        return

    thickness = max(3, round(min(image.size) / 230))
    draw.rectangle((x1, y1, x2, y2), outline=color, width=thickness)

    margin = 5
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_width, text_height = right - left, bottom - top
    tag_width = text_width + 2 * margin
    tag_height = text_height + 2 * margin
    tag_x = max(0, min(x1, image.width - tag_width))
    tag_y = y1 - tag_height if y1 >= tag_height else min(y1 + thickness, image.height - tag_height)
    tag_y = max(0, tag_y)
    draw.rectangle((tag_x, tag_y, tag_x + tag_width, tag_y + tag_height), fill=color)
    draw.text((tag_x + margin - left, tag_y + margin - top), label,
              font=font, fill="white", stroke_width=0)


def _annotate(image: Image.Image, detections: list[dict],
              ground_truth: list[dict], show_scores: bool):
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = _label_font(output.width, output.height)
    # Green = ground truth; model predictions use class-specific colors.
    for gt in ground_truth:
        _draw_box_label(draw, output, gt["xyxy"], f"GT {gt['name']}",
                        (20, 160, 40), font)
    for d in detections:
        if d["status"] not in ("accepted", "normal"):
            continue
        color = COLORS[d["class_id"] % len(COLORS)]
        label = d["name"]
        if show_scores:
            label += f" {d['confidence'] * 100:.1f}%"
        if d["status"] == "accepted":
            label += " (reviewed)"
        _draw_box_label(draw, output, d["xyxy"], label, color, font)
    return output


def _export_csv(parent: tk.Misc, rows: list[dict], report: dict):
    path = filedialog.asksaveasfilename(
        parent=parent, title="Save YOLO detection report", defaultextension=".csv",
        filetypes=[("CSV report", "*.csv")])
    if not path:
        return None
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "value"])
        for key, value in report.items():
            writer.writerow([key, value])
        writer.writerow([])
        writer.writerow(["id", "class", "confidence", "x1", "y1", "x2", "y2", "status"])
        for d in rows:
            writer.writerow([d["id"], d["name"], f"{d['confidence']:.4f}",
                             *(round(v, 2) for v in d["xyxy"]), d["status"]])
    return path


class _YOLOToolBase(ForensicsTool):
    requires_image = True
    model_path: Path
    is_blood_model = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        if document.current is None:
            messagebox.showwarning("No image", "Open an image first.", parent=parent)
            return None
        if self.is_blood_model and not self.model_path.is_file():
            messagebox.showerror("Missing model",
                                 f"Cannot find model:\n{self.model_path}\n\n"
                                 "Place your trained .pt file in the project's models folder.",
                                 parent=parent)
            return None
        try:
            # The standard pretrained weights can download on first use.
            model_source = (str(self.model_path) if self.model_path.is_file()
                            else "yolo11n.pt")
            names = _model_names(model_source)
        except Exception as exc:
            messagebox.showerror("YOLO load error", str(exc), parent=parent)
            return None

        config = _settings_dialog(parent, names, blood=self.is_blood_model)
        if config is None:
            return None
        image = document.current.convert("RGB")
        warning, sharpness, brightness = _image_quality(image)
        # Run at a lower confidence when doing manual review, to show candidates
        # that would otherwise be filtered out by the user's main threshold.
        inference_conf = 0.10 if config["review"] else config["confidence"]
        try:
            raw_boxes = _predict_safely(
                image, model_source, inference_conf, config["iou"]
            )
        except Exception as exc:
            messagebox.showerror("YOLO inference error", str(exc), parent=parent)
            return None

        detections = []
        for idx, box in enumerate(raw_boxes):
            class_id = box["class_id"]
            if class_id not in config["categories"]:
                continue
            score = box["confidence"]
            detections.append({
                "id": idx + 1, "class_id": class_id, "name": names[class_id],
                "confidence": score,
                "xyxy": tuple(box["xyxy"]),
                "status": "normal" if score >= config["confidence"] else "review",
            })

        if config["review"]:
            _review_dialog(parent, detections, names)
        ground_truth = []
        if config["gt_path"]:
            try:
                ground_truth = _read_ground_truth(
                    config["gt_path"], image.width, image.height, names)
            except (OSError, ValueError) as exc:
                messagebox.showwarning("Ground-truth error", str(exc), parent=parent)

        visible = [d for d in detections if d["status"] in ("normal", "accepted")]
        counts = Counter(d["name"] for d in visible)
        result_image = _annotate(image, detections, ground_truth, config["scores"])
        report = {
            "Model": self.model_path.name,
            "Confidence threshold": f"{config['confidence']:.2f}",
            "NMS IoU": f"{config['iou']:.2f}",
            "Total detected": len(visible),
            "Quality warning (heuristic)": warning,
            "Laplacian variance": f"{sharpness:.1f}",
            "Mean brightness": f"{brightness:.1f}",
            "Ground-truth boxes loaded": len(ground_truth),
        }
        for name in names.values():
            if self.is_blood_model or counts.get(name):
                report[f"{name} detected"] = counts.get(name, 0)

        # Individual scores in the Results panel, in image-detection order.
        # Limit the panel for densely populated microscope images; CSV contains all.
        panel_limit = 50
        for number, det in enumerate(visible[:panel_limit], start=1):
            report[f"Detection {number} · {det['name']}"] = (
                f"{det['confidence'] * 100:.1f}% confidence"
            )
        if len(visible) > panel_limit:
            report["Additional detections"] = (
                f"{len(visible) - panel_limit} more; export CSV for all confidence scores"
            )

        if config["export"]:
            try:
                saved_path = _export_csv(parent, detections, report)
                if saved_path:
                    report["CSV saved"] = saved_path
            except OSError as exc:
                messagebox.showerror("CSV export error", str(exc), parent=parent)

        return ToolResult(
            image=result_image,
            message=f"YOLO detected {len(visible)} object(s). {warning}.",
            details=report,
        )


class PretrainedYOLOTool(_YOLOToolBase):
    tool_id = "pretrained_yolo"
    title = "Pretrained YOLO Detection"
    category = "Set 9 · Object Detection"
    description = "Find everyday objects with pretrained YOLO11n."
    model_path = GENERAL_MODEL


class BloodCellDetectionTool(_YOLOToolBase):
    """Set 9.3 blood-smear studio, with manual review and actual comparisons."""
    tool_id = "blood_cell_detection"
    title = "Blood Cell Investigation Studio"
    category = "Set 9 · Object Detection"
    description = "Inspect and correct detections, compare preprocessing, export a report."
    model_path = BLOOD_MODEL
    is_blood_model = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        from .blood_studio import open_blood_studio
        return open_blood_studio(parent, document, self.model_path)
