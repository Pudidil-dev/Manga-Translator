"""
Optimized manga translation pipeline with parallel processing.
Designed for Intel CPU (i7 Gen 11) with threading optimization.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

from services import Services, OCR_WORKERS
from detect_bubbles import detect_bubbles
from process_bubble import process_bubble

logger = logging.getLogger(__name__)


class MangaPipeline:
    """
    Optimized manga translation pipeline.
    Uses singleton services and parallel processing.
    """

    def __init__(
        self,
        source_lang: str = "ja",
        target_lang: str = "id",
        font_path: str = "fonts/animeace_i.ttf",
        ocr_workers: int = OCR_WORKERS,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.font_path = font_path
        self.ocr_workers = ocr_workers

        # Font settings
        self.line_height = 16
        self.font_size = 14
        self.wrapping_ratio = 0.075

    def translate_image(self, image: Image.Image) -> Image.Image:
        """
        Translate all text bubbles in a manga image.

        Args:
            image: PIL Image input

        Returns:
            PIL Image with translated text
        """
        # Get singleton services - use hybrid OCR for multi-language support
        hybrid_ocr = Services.get_hybrid_ocr()
        translator = Services.get_translator()

        # Convert to numpy array
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Detect bubbles (uses singleton YOLO)
        detections = detect_bubbles(image)

        if not detections:
            logger.info("No bubbles detected")
            return image

        logger.info(f"Detected {len(detections)} bubbles")

        # Extract crops for OCR
        crops = []
        coords = []
        for det in detections:
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            crop = img_array[y1:y2, x1:x2].copy()
            crops.append(crop)
            coords.append((x1, y1, x2, y2))

        # Parallel OCR with hybrid engine
        texts = self._parallel_ocr(crops, hybrid_ocr, self.source_lang)
        logger.info(f"OCR completed: {len(texts)} texts extracted")

        # Filter empty texts
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not valid_indices:
            logger.info("No text found in bubbles")
            return image

        valid_texts = [texts[i] for i in valid_indices]
        valid_crops = [crops[i] for i in valid_indices]
        valid_coords = [coords[i] for i in valid_indices]

        # Batch translate
        translations = translator.translate_batch(
            valid_texts,
            source_lang=self.source_lang,
            target_lang=self.target_lang
        )
        logger.info(f"Translation completed: {len(translations)} texts translated")

        # Parallel render
        rendered = self._parallel_render(valid_crops, translations)

        # Apply to image
        for i, (x1, y1, x2, y2) in enumerate(valid_coords):
            img_array[y1:y2, x1:x2] = rendered[i]

        # Convert back to PIL
        result = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)

    def _parallel_ocr(self, crops: List[np.ndarray], hybrid_ocr, source_lang: str) -> List[str]:
        """
        Run OCR on multiple crops in parallel using hybrid OCR.
        """
        def ocr_one(crop):
            try:
                # Convert BGR to RGB for PIL
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                return hybrid_ocr.read(pil_img, lang=source_lang)
            except Exception as e:
                logger.warning(f"OCR failed: {e}")
                return ""

        texts = [None] * len(crops)

        with ThreadPoolExecutor(max_workers=self.ocr_workers) as executor:
            futures = {executor.submit(ocr_one, crop): i for i, crop in enumerate(crops)}

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    texts[idx] = future.result()
                except Exception as e:
                    logger.warning(f"OCR worker failed: {e}")
                    texts[idx] = ""

        return texts

    def _parallel_render(self, crops: List[np.ndarray], translations: List[str]) -> List[np.ndarray]:
        """
        Process bubbles and render text in parallel.
        """
        def render_one(args):
            crop, text = args
            try:
                # Process bubble (whiten)
                processed, contour = process_bubble(crop.copy())
                # Add text
                result = self._add_text(processed, text, contour)
                return result
            except Exception as e:
                logger.warning(f"Render failed: {e}")
                return crop

        results = [None] * len(crops)

        with ThreadPoolExecutor(max_workers=self.ocr_workers) as executor:
            futures = {
                executor.submit(render_one, (crops[i], translations[i])): i
                for i in range(len(crops))
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Render worker failed: {e}")
                    results[idx] = crops[idx]

        return results

    def _add_text(self, image: np.ndarray, text: str, contour: np.ndarray) -> np.ndarray:
        """
        Add translated text inside bubble contour.
        """
        from PIL import ImageDraw, ImageFont
        import textwrap

        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        x, y, w, h = cv2.boundingRect(contour)

        line_height = self.line_height
        font_size = self.font_size
        wrapping_ratio = self.wrapping_ratio

        # Wrap text
        wrapped_text = textwrap.fill(text, width=int(w * wrapping_ratio), break_long_words=True)
        font = ImageFont.truetype(self.font_path, size=font_size)

        lines = wrapped_text.split('\n')
        total_text_height = len(lines) * line_height

        # Shrink if needed
        while total_text_height > h and font_size > 6:
            line_height -= 2
            font_size -= 2
            wrapping_ratio += 0.025

            wrapped_text = textwrap.fill(text, width=int(w * wrapping_ratio), break_long_words=True)
            font = ImageFont.truetype(self.font_path, size=font_size)

            lines = wrapped_text.split('\n')
            total_text_height = len(lines) * line_height

        # Vertical centering
        text_y = y + (h - total_text_height) // 2

        for line in lines:
            text_length = draw.textlength(line, font=font)
            text_x = x + (w - text_length) // 2
            draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))
            text_y += line_height

        # Convert back
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def translate_manga(
    image_path: str,
    output_path: Optional[str] = None,
    source_lang: str = "ja",
    target_lang: str = "id",
    font_path: str = "fonts/animeace_i.ttf",
) -> Image.Image:
    """
    Convenience function to translate a manga image.

    Args:
        image_path: Path to input image
        output_path: Optional path to save result
        source_lang: Source language code
        target_lang: Target language code
        font_path: Path to font file

    Returns:
        Translated PIL Image
    """
    pipeline = MangaPipeline(
        source_lang=source_lang,
        target_lang=target_lang,
        font_path=font_path,
    )

    image = Image.open(image_path)
    result = pipeline.translate_image(image)

    if output_path:
        result.save(output_path)
        logger.info(f"Saved to: {output_path}")

    return result


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path> [output_path]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Preload models
    print("Loading models...")
    Services.preload_all()

    # Translate
    print(f"Processing: {input_path}")
    result = translate_manga(input_path, output_path)

    if not output_path:
        result.show()
