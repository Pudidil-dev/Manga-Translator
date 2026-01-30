"""Add translated text to speech bubbles."""

from typing import Union
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
import cv2


def add_text(
    image: np.ndarray,
    text: str,
    font_path: Union[str, Path],
    bubble_contour: np.ndarray
) -> np.ndarray:
    """
    Add text inside a speech bubble contour.

    Args:
        image: Processed bubble image (cv2 format - BGR).
        text: Text to be placed inside the speech bubble.
        font_path: Path to font file.
        bubble_contour: Contour of the detected speech bubble.

    Returns:
        Image with text placed inside the speech bubble.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    x, y, w, h = cv2.boundingRect(bubble_contour)

    # Shrink text box slightly to avoid touching/overflowing bubble edges
    pad = max(4, int(min(w, h) * 0.06))
    if w - 2 * pad > 10 and h - 2 * pad > 10:
        x += pad
        y += pad
        w -= 2 * pad
        h -= 2 * pad

    line_height = 16
    font_size = 14
    min_font_size = 8
    wrapping_ratio = 0.075

    while True:
        wrapped_text = textwrap.fill(text, width=int(w * wrapping_ratio),
                                     break_long_words=False)

        font = ImageFont.truetype(font_path, size=font_size)
        lines = wrapped_text.split('\n') if wrapped_text else []
        total_text_height = len(lines) * line_height
        max_line_width = max((draw.textlength(line, font=font) for line in lines), default=0)

        if total_text_height <= h and max_line_width <= w:
            break

        if font_size <= min_font_size:
            break

        font_size -= 1
        line_height = max(8, int(font_size * 1.2))
        wrapping_ratio += 0.01

    # Vertical centering
    text_y = y + (h - total_text_height) // 2

    for line in lines:
        text_length = draw.textlength(line, font=font)

        # Horizontal centering
        text_x = x + (w - text_length) // 2

        draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))

        text_y += line_height

    image[:, :, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image
