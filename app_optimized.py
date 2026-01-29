"""
Optimized Manga Translator - Gradio Interface
Uses singleton services and parallel processing for faster translation.
"""

import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from PIL import Image

from services import Services
from pipeline import MangaPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
EXAMPLE_LIST = [["examples/0.png"], ["examples/ex0.png"]]
TITLE = "Manga Translator (Optimized)"
DESCRIPTION = """
Translate text in manga bubbles with optimized pipeline.
- Singleton model loading (no reload per image)
- Parallel OCR processing
- Batch translation with caching
"""

# Available languages
LANGUAGES = [
    ("Indonesian", "id"),
    ("English", "en"),
    ("Chinese (Simplified)", "zh"),
    ("Korean", "ko"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Thai", "th"),
    ("Vietnamese", "vi"),
]

# Available fonts
FONTS = [
    ("Anime Ace", "fonts/animeace_i.ttf"),
    ("Manga Temple", "fonts/mangati.ttf"),
    ("Arial Italic", "fonts/ariali.ttf"),
]


def predict(img, target_lang: str, font: str):
    """
    Translate manga image using optimized pipeline.
    """
    if img is None:
        return None

    if target_lang is None:
        target_lang = "id"
    if font is None:
        font = "fonts/animeace_i.ttf"

    try:
        pipeline = MangaPipeline(
            source_lang="ja",
            target_lang=target_lang,
            font_path=font,
        )

        result = pipeline.translate_image(img)
        return result

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise gr.Error(f"Translation failed: {str(e)}")


def preload_models():
    """Preload all models at startup."""
    logger.info("Preloading models (this may take a moment)...")
    Services.preload_all()
    logger.info("Models loaded successfully!")


# Build interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Input Manga Image"),
        gr.Dropdown(
            choices=LANGUAGES,
            label="Target Language",
            value="id",
        ),
        gr.Dropdown(
            choices=FONTS,
            label="Font",
            value="fonts/animeace_i.ttf",
        ),
    ],
    outputs=[gr.Image(type="pil", label="Translated Image")],
    examples=EXAMPLE_LIST,
    title=TITLE,
    description=DESCRIPTION,
    cache_examples=False,
)

if __name__ == "__main__":
    # Preload models before launching
    preload_models()

    demo.launch(
        debug=False,
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )
