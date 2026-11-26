
"""Set 10.1 — CLIP Zero-Shot Classification."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import textwrap
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from PIL import Image

from ..base import ForensicsTool, ToolResult
from ..common import MODELS_FOLDER


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "openai/clip-vit-base-patch32"
CACHE_DIR = MODELS_FOLDER / "clip"

CLASS_PRESETS = {
    "Custom": ["", "", ""],
    "Vehicles": ["car", "bus", "truck", "motorcycle", "bicycle"],
    "Animals": ["cat", "dog", "horse", "rabbit"],
    "Beverages": ["coffee", "tea", "soup", "hot chocolate"],
}

PROMPT_TEMPLATES = {
    "As written": "{}",
    "Photograph (singular)": "a photo of a {}.",
    "Photograph (general)": "a photo of {}.",
    "Close-up": "a close-up photo of {}.",
    "Surveillance": "a surveillance image showing {}.",
    "Blurry image": "a blurry photo of {}.",
    "Illustration": "an illustration of {}.",
}

_model = None
_processor = None


# ---------------------------------------------------------
# 1. Load pretrained CLIP
# ---------------------------------------------------------

def load_clip_model():
    global _model, _processor

    import torch
    from transformers import CLIPModel, CLIPProcessor

    if _model is None:
        torch.set_num_threads(1)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        print("[CLIP] Loading model...", flush=True)

        model = CLIPModel.from_pretrained(
            MODEL_NAME,
            cache_dir=str(CACHE_DIR),
            attn_implementation="eager",
        ).to("cpu").eval()

        processor = CLIPProcessor.from_pretrained(
            MODEL_NAME,
            cache_dir=str(CACHE_DIR),
            use_fast=False,
        )

        _model, _processor = model, processor

    return _model, _processor


# ---------------------------------------------------------
# 2. Image and text embeddings
# ---------------------------------------------------------

def clip_embed_images(images):
    import torch
    import torch.nn.functional as F

    model, processor = load_clip_model()

    inputs = processor(
        images=images,
        return_tensors="pt",
    ).to("cpu")

    print("[CLIP] Encoding image...", flush=True)

    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

        if not torch.is_tensor(embeddings):
            embeddings = embeddings.pooler_output

        return F.normalize(embeddings, dim=-1)


def clip_embed_texts(texts):
    import torch
    import torch.nn.functional as F

    model, processor = load_clip_model()

    # Validate CLIP's maximum text length.
    limit = model.config.text_config.max_position_embeddings
    tokens = processor.tokenizer(texts, truncation=False)

    if any(len(t) > limit for t in tokens["input_ids"]):
        raise ValueError(
            f"Each prompt must contain at most {limit} tokens."
        )

    inputs = processor(
        text=texts,
        padding=True,
        return_tensors="pt",
    ).to("cpu")

    print("[CLIP] Encoding text...", flush=True)

    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)

        if not torch.is_tensor(embeddings):
            embeddings = embeddings.pooler_output

        return F.normalize(embeddings, dim=-1)


# ---------------------------------------------------------
# 3. Zero-shot classification
# ---------------------------------------------------------

def zero_shot(image, labels, template="a photo of a {}."):


    import torch

    model, _ = load_clip_model()

    prompts = [
        template.replace("{}", label)
        for label in labels
    ]

    img_emb = clip_embed_images([image])
    txt_emb = clip_embed_texts(prompts)

    with torch.no_grad():

        # Cosine similarity of normalized embeddings.
        similarities = img_emb @ txt_emb.T

        # Apply CLIP's learned scaling factor.
        logits = model.logit_scale.exp() * similarities

        # Convert scores into relative probabilities.
        probabilities = logits.softmax(dim=-1)

    return (
        probabilities[0].float().cpu().numpy(),
        similarities[0].float().cpu().numpy(),
    )

def show_zero_shot(image, labels, probabilities, template):

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    order = np.argsort(probabilities)
    best = int(np.argmax(probabilities))

    names = [
        textwrap.fill(labels[i], width=28)
        for i in order
    ]

    height = max(
        3.2,
        sum(name.count("\n") + 1 for name in names) * 0.4 + 1.5,
    )

    fig = Figure(
        figsize=(10, height),
        facecolor="white",
        layout="constrained",
    )

    FigureCanvasAgg(fig)

    a0, a1 = fig.subplots(
        1, 2,
        gridspec_kw={"width_ratios": [1, 1.4]},
    )

    # Original image.
    a0.imshow(image)
    a0.axis("off")
    a0.set_title(
        textwrap.fill(labels[best], 30),
        fontsize=11,
    )

    # Horizontal probability chart.
    positions = np.arange(len(labels))

    a1.barh(
        positions,
        probabilities[order],
        color="#3b6ea8",
        height=0.5,
    )

    a1.set_yticks(positions, names)
    a1.set_xlim(0, 1)
    a1.set_xlabel("probability")

    a1.set_title(
        textwrap.fill(f'template: "{template}"', 48),
        fontsize=11,
    )

    # Display probabilities with four decimal places.
    for i, value in enumerate(probabilities[order]):

        if value > 0.85:
            x = value - 0.02
            alignment = "right"
            color = "white"
        else:
            x = value + 0.02
            alignment = "left"
            color = "black"

        a1.text(
            x,
            i,
            f"{value:.4f}",
            va="center",
            ha=alignment,
            color=color,
            fontsize=10,
        )

    # Convert the figure to a PIL image.
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format="png", dpi=150)
        buffer.seek(0)

        with Image.open(buffer) as output:
            return output.convert("RGB")


# ---------------------------------------------------------
# 5. Configuration window
# ---------------------------------------------------------

def ask_configuration(parent):

    window = tk.Toplevel(parent)
    window.title("CLIP Classification")
    window.geometry("540x680")
    window.minsize(440, 420)
    window.transient(parent.winfo_toplevel())

    BG = "#ffffff"
    TEXT = "#1e293b"
    MUTED = "#64748b"
    BORDER = "#cbd5e1"
    SURFACE = "#f8fafc"
    BLUE = "#2563eb"

    window.configure(bg=BG)

    result = None
    entries = []
    updating_fields = False

    # Reusable styled widgets.
    def label(container, text, size=11, bold=False, color=TEXT):
        return tk.Label(
            container,
            text=text,
            font=("Helvetica", size, "bold" if bold else "normal"),
            bg=BG,
            fg=color,
            anchor="w",
            justify="left",
            wraplength=440,
        )

    def entry(container, variable):
        return tk.Entry(
            container,
            textvariable=variable,
            font=("Helvetica", 12),
            bg=SURFACE,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=BLUE,
        )

    def button(container, text, command, primary=False):

        background = BLUE if primary else SURFACE
        foreground = "white" if primary else TEXT

        widget = tk.Label(
            container,
            text=text,
            bg=background,
            fg=foreground,
            font=("Helvetica", 11, "bold"),
            padx=14,
            pady=9,
            cursor="hand2",
            takefocus=True,
            highlightthickness=1,
            highlightbackground=background,
        )

        widget.bind("<Button-1>", lambda e: command())
        widget.bind("<Return>", lambda e: command())
        widget.bind("<space>", lambda e: command())

        widget.bind(
            "<FocusIn>",
            lambda e: widget.configure(highlightbackground=BLUE),
        )

        widget.bind(
            "<FocusOut>",
            lambda e: widget.configure(highlightbackground=background),
        )

        return widget

    # Header.
    header = tk.Frame(window, bg=BG)
    header.pack(fill="x", padx=24, pady=(20, 12))

    label(
        header, "Zero-shot classification", 20, True
    ).pack(fill="x")

    label(
        header,
        "Compare the currently opened image with custom descriptions.",
        size=10,
        color=MUTED,
    ).pack(fill="x", pady=(5, 0))

    # Fixed footer.
    footer = tk.Frame(window, bg=BG)
    footer.pack(side="bottom", fill="x", padx=24, pady=16)

    # Scrollable content.
    area = tk.Frame(window, bg=BG)
    area.pack(fill="both", expand=True, padx=24)

    canvas = tk.Canvas(
        area, bg=BG, highlightthickness=0
    )

    scrollbar = ttk.Scrollbar(
        area, orient="vertical", command=canvas.yview
    )

    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    body = tk.Frame(canvas, bg=BG)

    body_id = canvas.create_window(
        (0, 0), window=body, anchor="nw"
    )

    body.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        ),
    )

    canvas.bind(
        "<Configure>",
        lambda e: canvas.itemconfigure(
            body_id, width=e.width
        ),
    )

    # -----------------------------------------------------
    # Class presets
    # -----------------------------------------------------

    label(
        body, "Class preset", bold=True
    ).pack(fill="x", pady=(8, 6))

    preset = tk.StringVar(master=window, value="Vehicles")

    def apply_preset(event=None):
        names = CLASS_PRESETS[preset.get()]
        count.set(str(len(names)))
        create_fields(values=names)

    preset_menu = ttk.Combobox(
        body,
        textvariable=preset,
        values=list(CLASS_PRESETS.keys()),
        state="readonly",
        font=("Helvetica", 11),
    )

    preset_menu.pack(fill="x", ipady=4)
    preset_menu.bind("<<ComboboxSelected>>", apply_preset)

    label(
        body,
        "Choose a preset or define your own classes.",
        size=10,
        color=MUTED,
    ).pack(fill="x", pady=(4, 0))

    # -----------------------------------------------------
    # Arbitrary number of classes
    # -----------------------------------------------------

    label(
        body, "Number of candidate classes", bold=True
    ).pack(fill="x", pady=(16, 6))

    count = tk.StringVar(
        master=window,
        value=str(len(CLASS_PRESETS["Vehicles"])),
    )

    count_row = tk.Frame(body, bg=BG)
    count_row.pack(fill="x")

    def change_count(amount):
        try:
            n = int(count.get())
        except ValueError:
            n = len(entries) or 3

        count.set(str(min(100, max(2, n + amount))))
        create_fields()

    button(
        count_row, "−", lambda: change_count(-1)
    ).pack(side="left")

    count_entry = entry(count_row, count)
    count_entry.configure(justify="center")

    count_entry.pack(
        side="left",
        fill="x",
        expand=True,
        padx=8,
        ipady=7,
    )

    button(
        count_row, "+", lambda: change_count(1)
    ).pack(side="right")

    # -----------------------------------------------------
    # Arbitrary text for each class
    # -----------------------------------------------------

    label(
        body, "Class descriptions", bold=True
    ).pack(fill="x", pady=(16, 4))

    label(
        body,
        "Enter a word, phrase or complete sentence.",
        size=10,
        color=MUTED,
    ).pack(fill="x")

    fields = tk.Frame(body, bg=BG)
    fields.pack(fill="x", pady=6)

    def get_labels():
        return [var.get().strip() for var in entries]

    def class_changed(*args):

        if not updating_fields:

            # A modified preset becomes a custom class list.
            if get_labels() != CLASS_PRESETS[preset.get()]:
                preset.set("Custom")

            refresh_preview()

    def create_fields(event=None, values=None):

        nonlocal updating_fields

        try:
            n = int(count.get())

            if not 2 <= n <= 100:
                return

        except ValueError:
            return

        if n == len(entries) and values is None:
            return

        previous = (
            list(values)
            if values is not None
            else get_labels()
        )

        updating_fields = True

        for widget in fields.winfo_children():
            widget.destroy()

        entries.clear()

        for i in range(n):

            row = tk.Frame(fields, bg=BG)
            row.pack(fill="x", pady=4)

            label(
                row,
                f"{i + 1:02d}",
                color=MUTED,
            ).pack(side="left", padx=(0, 10))

            var = tk.StringVar(
                master=window,
                value=previous[i] if i < len(previous) else "",
            )

            entry(row, var).pack(
                side="left",
                fill="x",
                expand=True,
                ipady=7,
            )

            entries.append(var)
            var.trace_add("write", class_changed)

        updating_fields = False

        if values is None:
            preset.set("Custom")

        refresh_preview()

    count_entry.bind("<Return>", create_fields)
    count_entry.bind("<FocusOut>", create_fields)

    # -----------------------------------------------------
    # Prompt template
    # -----------------------------------------------------

    label(
        body, "Prompt style", bold=True
    ).pack(fill="x", pady=(16, 6))

    style = tk.StringVar(
        master=window,
        value="Photograph (general)",
    )

    custom = tk.StringVar(
        master=window,
        value="a photo of {}.",
    )

    def get_template():
        if style.get() == "Custom":
            return custom.get().strip()

        return PROMPT_TEMPLATES[style.get()]

    def change_style(event=None):
        if style.get() == "Custom":
            custom_entry.pack(fill="x", ipady=7, pady=6)
        else:
            custom_entry.pack_forget()

        refresh_preview()

    menu = ttk.Combobox(
        body,
        textvariable=style,
        values=[*PROMPT_TEMPLATES.keys(), "Custom"],
        state="readonly",
        font=("Helvetica", 11),
    )

    menu.pack(fill="x", ipady=4)
    menu.bind("<<ComboboxSelected>>", change_style)

    custom_frame = tk.Frame(body, bg=BG)
    custom_frame.pack(fill="x")

    custom_entry = entry(custom_frame, custom)

    # -----------------------------------------------------
    # Live prompt preview
    # -----------------------------------------------------

    label(
        body, "Generated prompts", bold=True
    ).pack(fill="x", pady=(14, 6))

    preview = tk.Label(
        body,
        bg=SURFACE,
        fg=MUTED,
        font=("Helvetica", 11),
        justify="left",
        anchor="w",
        wraplength=410,
        padx=12,
        pady=10,
    )

    preview.pack(fill="x")

    def refresh_preview(*args):

        prompts = [
            get_template().replace("{}", name)
            for name in get_labels() if name
        ]

        display = "\n".join(prompts[:4])

        if len(prompts) > 4:
            display += f"\n+ {len(prompts) - 4} more"

        preview.configure(
            text=display or "Your prompts will appear here."
        )

    custom.trace_add("write", refresh_preview)

    # -----------------------------------------------------
    # Optional ground truth
    # -----------------------------------------------------

    ground = tk.StringVar(master=window)
    evaluation = tk.Frame(body, bg=BG)

    def toggle_ground():

        if evaluation.winfo_manager():

            evaluation.pack_forget()
            ground.set("")
            toggle.configure(text="+ Add ground truth")

        else:

            evaluation.pack(fill="x", pady=8)
            toggle.configure(text="− Hide ground truth")

    toggle = button(
        body, "+ Add ground truth", toggle_ground
    )

    toggle.pack(anchor="w", pady=(14, 4))

    entry(evaluation, ground).pack(
        fill="x", ipady=7
    )

    label(
        evaluation,
        "Optional. Enter the known correct class.",
        size=10,
        color=MUTED,
    ).pack(fill="x", pady=5)

    # -----------------------------------------------------
    # Validate configuration
    # -----------------------------------------------------

    def submit():
        nonlocal result

        labels = get_labels()
        template = get_template()

        try:
            n = int(count.get())

            valid_count = (
                2 <= n <= 100
                and n == len(labels)
            )

        except ValueError:
            valid_count = False

        if not valid_count:
            messagebox.showwarning(
                "CLIP",
                "Select between 2 and 100 classes.",
                parent=window,
            )
            return

        if any(not name for name in labels):
            messagebox.showwarning(
                "CLIP",
                "Complete every class description.",
                parent=window,
            )
            return

        if len({name.casefold() for name in labels}) != len(labels):
            messagebox.showwarning(
                "CLIP",
                "Class descriptions must be unique.",
                parent=window,
            )
            return

        if "{}" not in template:
            messagebox.showwarning(
                "CLIP",
                "The custom template must contain {}.",
                parent=window,
            )
            return

        result = labels, template, ground.get().strip()
        window.destroy()

    # Footer actions.
    button(
        footer, "Cancel", window.destroy
    ).pack(side="left", padx=(0, 10))

    button(
        footer, "Classify image", submit, primary=True
    ).pack(side="right", fill="x", expand=True)

    # Load the default Vehicles preset.
    apply_preset()

    window.bind("<Escape>", lambda e: window.destroy())
    window.wait_visibility()
    window.grab_set()

    parent.wait_window(window)

    return result


# ---------------------------------------------------------
# 6. Connect to the Forensics App
# ---------------------------------------------------------

class CLIPZeroShotTool(ForensicsTool):

    tool_id = "clip_zero_shot"
    title = "CLIP Zero-Shot Classification"
    category = "Set 10 · Vision Foundation Models"

    description = (
        "Classify images using pretrained CLIP with an "
        "arbitrary number of classes and text descriptions."
    )

    requires_image = True

    def run(self, parent, document):

        try:

            if document.current is None:
                messagebox.showwarning(
                    "CLIP",
                    "Open an image first.",
                    parent=parent,
                )
                return None

            configuration = ask_configuration(parent)

            if configuration is None:
                return None

            labels, template, ground_truth = configuration

            # Use the image opened through the main toolbar.
            image = document.current.convert("RGB")

            # Perform zero-shot classification.
            probabilities, similarities = zero_shot(
                image, labels, template
            )
            
            output = show_zero_shot(
                image, labels, probabilities, template
            )

        except Exception as exc:

            messagebox.showerror(
                "CLIP Error",
                str(exc),
                parent=parent,
            )
            return None

        best = int(np.argmax(probabilities))
        prediction = labels[best]

        details = {
            "Model": "CLIP ViT-B/32",
            "Classes": len(labels),
            "Template": template,
            "Prediction": prediction,
            "Probability": f"{probabilities[best]:.2%}",
            "Similarity": f"{similarities[best]:.4f}",
            "Ground truth": ground_truth or "Not provided",
        }

        # Optional ground truth evaluation.
        if ground_truth:
            details["Evaluation"] = (
                "Correct"
                if prediction.casefold() == ground_truth.casefold()
                else "Incorrect"
            )

        # Scores for each candidate class.
        for rank, i in enumerate(
            np.argsort(probabilities)[::-1], start=1
        ):

            details[f"{rank}. {labels[i]}"] = (
                f"{probabilities[i]:.2%}"
            )

            details[f"Similarity {rank}"] = (
                f"{similarities[i]:.4f}"
            )

        details["Note"] = (
            "Probabilities are relative to the candidate classes."
        )

        return ToolResult(
            preview=output,
            message=f"Prediction: {prediction}",
            details=details,
        )
