"""Code shared by tools from different sets.

Code shared only inside one set lives in that set's common.py (for example set4/common.py).
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# File types offered by every "open image" dialog
IMAGE_TYPES = [
    ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
    ("All files", "*.*"),
]

SEPARATOR = 4
# Title font size = panel width / FONT_RATIO
# Make the titles readable for small and large images
FONT_RATIO = 10
MIN_FONT = 16


# Draw a canvas with titles, then put same-size images together (Set 2.1 channels, Set 4.1)
def compose_side_by_side(panels: dict[str, Image.Image]) -> Image.Image:
    first = next(iter(panels.values()))  # Get first pic
    width, height = first.size
    count = len(panels)

    font_size = max(MIN_FONT, width // FONT_RATIO)
    label_height = font_size * 3 // 2  # Room for the text plus a small margin
    margin = font_size // 4

    canvas = Image.new(  # Use the first image to draw canvas
        first.mode,
        (width * count + SEPARATOR * (count - 1), height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(font_size)

    for index, (name, panel) in enumerate(panels.items()):  # Draw each panel
        x = index * (width + SEPARATOR)
        canvas.paste(panel, (x, label_height))
        draw.text((x + margin, margin), name, fill="black", font=font)
    return canvas
