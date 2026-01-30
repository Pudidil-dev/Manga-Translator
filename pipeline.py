"""
Optimized manga translation pipeline with parallel processing.
Designed for Intel CPU (i7 Gen 11) with threading optimization.

Combined Architecture Features:
- Mode-based processing (realtime/quality/premium)
- OSB (Outside Speech Bubble) detection
- Advanced SAM2 segmentation for precise masks
- Advanced inpainting (canvas/opencv/flux)
- Dual YOLO detection with conjoined bubble support

Ported from MangaTranslator's advanced detection/cleaning for higher accuracy.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Literal, Dict, Any

import cv2
import numpy as np
from PIL import Image

from services import Services, OCR_WORKERS
from detect_bubbles import detect_bubbles
from process_bubble import process_bubble
from config import ProcessingConfig

# Advanced detection/cleaning modules (ported from MangaTranslator)
try:
    from advanced_detection import detect_speech_bubbles, get_advanced_detector
    ADVANCED_DETECTION_AVAILABLE = True
except ImportError:
    ADVANCED_DETECTION_AVAILABLE = False

try:
    from advanced_cleaning import clean_speech_bubbles, get_advanced_cleaner
    ADVANCED_CLEANING_AVAILABLE = True
except ImportError:
    ADVANCED_CLEANING_AVAILABLE = False

logger = logging.getLogger(__name__)


class MangaPipeline:
    """
    Optimized manga translation pipeline.
    Uses singleton services and parallel processing.

    Combined Architecture Support:
    - mode: "realtime" (<1s), "quality" (2-3s), "premium" (5-10s)
    - detect_osb: Detect text outside speech bubbles
    - inpaint_method: "canvas", "opencv", or "flux"
    - use_advanced: Use advanced SAM2 detection and cleaning (ported from MangaTranslator)
    """

    def __init__(
        self,
        source_lang: str = "ja",
        target_lang: str = "id",
        font_path: str = "fonts/animeace_i.ttf",
        ocr_workers: int = OCR_WORKERS,
        # Combined Architecture options
        mode: Literal["realtime", "quality", "premium"] = "realtime",
        detect_osb: bool = True,
        inpaint_method: Literal["auto", "canvas", "opencv", "flux"] = "auto",
        config: ProcessingConfig = None,
        # Advanced options (ported from MangaTranslator)
        use_advanced: bool = True,  # Use advanced SAM2 detection
        use_sam: bool = True,  # Use SAM for mask refinement
        conjoined_detection: bool = True,  # Detect conjoined bubbles
        osb_text_verification: bool = False,  # Expand boxes with OSB text
        inpaint_colored: bool = False,  # AI inpainting for colored bubbles
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.font_path = font_path
        self.ocr_workers = ocr_workers

        # Mode and options
        self.mode = mode
        self.detect_osb = detect_osb
        self.inpaint_method = inpaint_method
        self.config = config or ProcessingConfig.from_mode(mode)

        # Advanced options
        self.use_advanced = use_advanced and ADVANCED_DETECTION_AVAILABLE
        self.use_sam = use_sam and (mode != "realtime")  # SAM only for quality/premium
        self.conjoined_detection = conjoined_detection
        self.osb_text_verification = osb_text_verification
        self.inpaint_colored = inpaint_colored and (mode == "premium")

        # Font settings
        self.line_height = 16
        self.font_size = 14
        self.wrapping_ratio = 0.075

        # Timing info
        self._timing = {}

    def translate_image(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """
        Translate all text bubbles in a manga image.

        Args:
            image: PIL Image input

        Returns:
            Tuple of (translated PIL Image, timing dict)
        """
        self._timing = {"mode": self.mode, "advanced_detection": self.use_advanced}
        total_start = time.time()

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

        # =====================================================================
        # STEP 1: Bubble Detection (Advanced or Simple)
        # =====================================================================
        t0 = time.time()

        if self.use_advanced and ADVANCED_DETECTION_AVAILABLE:
            # Use advanced detection with SAM2 and dual YOLO
            logger.info("Using advanced detection (SAM2 + dual YOLO)")
            detections = detect_speech_bubbles(
                image=image,
                confidence=self.config.detection.bubble_confidence,
                conjoined_confidence=self.config.detection.conjoined_confidence,
                use_sam=self.use_sam,
                conjoined_detection=self.conjoined_detection,
                osb_text_verification=self.osb_text_verification,
                verbose=logger.isEnabledFor(logging.DEBUG)
            )
            self._timing["detection_type"] = "advanced_sam2"
        else:
            # Use simple detection
            raw_detections = detect_bubbles(image)
            # Convert to dict format
            detections = []
            for det in raw_detections:
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(det[4]) if len(det) > 4 else 1.0,
                    "class": "speech_bubble",
                    "sam_mask": None
                })
            self._timing["detection_type"] = "simple_yolo"

        self._timing["detection_ms"] = int((time.time() - t0) * 1000)
        logger.info(f"Detected {len(detections)} bubbles")

        # =====================================================================
        # STEP 2: OSB Detection (if enabled)
        # =====================================================================
        osb_detections = []
        if self.detect_osb:
            t0 = time.time()
            osb_detections = self._detect_osb(image, detections)
            self._timing["osb_detection_ms"] = int((time.time() - t0) * 1000)
            logger.info(f"Detected {len(osb_detections)} OSB regions")

        # Combine all detections
        all_detections = list(detections) + osb_detections

        if not all_detections:
            logger.info("No text regions detected")
            self._timing["total_ms"] = int((time.time() - total_start) * 1000)
            return image, self._timing

        # =====================================================================
        # STEP 3: Advanced Cleaning (for quality/premium modes)
        # =====================================================================
        cleaned_image = img_array.copy()
        processed_bubbles = []

        if self.use_advanced and ADVANCED_CLEANING_AVAILABLE and self.mode != "realtime":
            t0 = time.time()
            logger.info(f"Using advanced cleaning (mode={self.mode})")

            # Convert to PIL for cleaning
            pil_for_cleaning = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

            cleaned_pil, processed_bubbles = clean_speech_bubbles(
                image=pil_for_cleaning,
                detections=all_detections,
                inpaint_method=self.inpaint_method,
                inpaint_colored=self.inpaint_colored,
                verbose=logger.isEnabledFor(logging.DEBUG),
                mode=self.mode
            )

            cleaned_image = cv2.cvtColor(np.array(cleaned_pil), cv2.COLOR_RGB2BGR)
            self._timing["cleaning_ms"] = int((time.time() - t0) * 1000)
            self._timing["cleaning_type"] = "advanced"
        elif self.mode != "realtime" or self.inpaint_method != "canvas":
            # Fallback to simple inpainting
            t0 = time.time()
            cleaned_image = self._inpaint(img_array, all_detections)
            self._timing["inpaint_ms"] = int((time.time() - t0) * 1000)
            self._timing["cleaning_type"] = "simple"

        # =====================================================================
        # STEP 4: Extract crops for OCR
        # =====================================================================
        crops = []
        coords = []
        region_types = []

        for det in all_detections:
            # Handle dict format from advanced detection
            if isinstance(det, dict):
                bbox = det.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = bbox
                else:
                    continue
                region_type = det.get("class", "bubble")
            elif hasattr(det, 'bbox'):
                x1, y1, x2, y2 = det.bbox
                region_type = getattr(det, 'region_type', 'bubble')
            else:
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                region_type = "bubble"

            # Use cleaned image for crop
            crop = cleaned_image[y1:y2, x1:x2].copy()
            crops.append(crop)
            coords.append((x1, y1, x2, y2))
            region_types.append(region_type)

        # =====================================================================
        # STEP 5: Parallel OCR
        # =====================================================================
        t0 = time.time()
        texts = self._parallel_ocr(crops, hybrid_ocr, self.source_lang)
        self._timing["ocr_ms"] = int((time.time() - t0) * 1000)
        logger.info(f"OCR completed: {len(texts)} texts extracted")

        # Filter empty texts
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not valid_indices:
            logger.info("No text found in regions")
            self._timing["total_ms"] = int((time.time() - total_start) * 1000)
            result = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            return Image.fromarray(result), self._timing

        valid_texts = [texts[i] for i in valid_indices]
        valid_crops = [crops[i] for i in valid_indices]
        valid_coords = [coords[i] for i in valid_indices]
        valid_types = [region_types[i] for i in valid_indices]

        # =====================================================================
        # STEP 6: Batch Translation
        # =====================================================================
        t0 = time.time()
        translations = translator.translate_batch(
            valid_texts,
            source_lang=self.source_lang,
            target_lang=self.target_lang
        )
        self._timing["translation_ms"] = int((time.time() - t0) * 1000)
        logger.info(f"Translation completed: {len(translations)} texts translated")

        # =====================================================================
        # STEP 7: Parallel Render
        # =====================================================================
        t0 = time.time()
        rendered = self._parallel_render(valid_crops, translations, valid_types)
        self._timing["render_ms"] = int((time.time() - t0) * 1000)

        # Apply to image
        for i, (x1, y1, x2, y2) in enumerate(valid_coords):
            cleaned_image[y1:y2, x1:x2] = rendered[i]

        # Convert back to PIL
        result = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
        self._timing["total_ms"] = int((time.time() - total_start) * 1000)

        return Image.fromarray(result), self._timing

    def _detect_osb(self, image: Image.Image, bubble_detections: list) -> list:
        """
        Detect text outside speech bubbles (SFX, narration, etc.).

        Args:
            image: Input PIL Image
            bubble_detections: Existing bubble detections to exclude

        Returns:
            List of OSB text regions
        """
        try:
            from osb_detection import get_hybrid_detector, TextRegion

            detector = get_hybrid_detector()
            bubble_regions, osb_regions = detector.detect_all(
                image,
                bubble_detections,
                detect_osb=True,
                osb_confidence=self.config.detection.osb_confidence
            )

            return osb_regions

        except ImportError:
            logger.warning("OSB detection module not available")
            return []
        except Exception as e:
            logger.warning(f"OSB detection failed: {e}")
            return []

    def _inpaint(self, img_array: np.ndarray, detections: list) -> np.ndarray:
        """
        Inpaint text regions before rendering.

        Args:
            img_array: Image as numpy array (BGR)
            detections: List of text regions to inpaint

        Returns:
            Inpainted image array
        """
        try:
            from inpainting import inpaint_image

            # Convert to PIL for inpainting
            rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            # Determine method based on mode
            if self.inpaint_method == "auto":
                if self.mode == "realtime":
                    method = "canvas"
                elif self.mode == "quality":
                    method = "opencv"
                else:
                    method = "flux"
            else:
                method = self.inpaint_method

            # Inpaint
            inpainted = inpaint_image(
                pil_image,
                detections,
                method=method,
                mode=self.mode
            )

            # Convert back to BGR
            return cv2.cvtColor(np.array(inpainted), cv2.COLOR_RGB2BGR)

        except ImportError:
            logger.warning("Inpainting module not available")
            return img_array
        except Exception as e:
            logger.warning(f"Inpainting failed: {e}")
            return img_array

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

    def _parallel_render(self, crops: List[np.ndarray], translations: List[str], region_types: List[str] = None) -> List[np.ndarray]:
        """
        Process bubbles and render text in parallel.

        Args:
            crops: List of image crops
            translations: List of translated texts
            region_types: List of region types ("bubble" or "osb")
        """
        if region_types is None:
            region_types = ["bubble"] * len(crops)

        def render_one(args):
            crop, text, region_type = args
            try:
                if region_type == "osb":
                    # OSB: Different rendering (e.g., with outline)
                    result = self._add_osb_text(crop, text)
                else:
                    # Regular bubble
                    processed, contour = process_bubble(crop.copy())
                    result = self._add_text(processed, text, contour)
                return result
            except Exception as e:
                logger.warning(f"Render failed: {e}")
                return crop

        results = [None] * len(crops)

        with ThreadPoolExecutor(max_workers=self.ocr_workers) as executor:
            futures = {
                executor.submit(render_one, (crops[i], translations[i], region_types[i])): i
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

    def _add_osb_text(self, image: np.ndarray, text: str) -> np.ndarray:
        """
        Add translated text for OSB regions (SFX, narration).
        Uses different styling than regular bubbles.
        """
        from PIL import ImageDraw, ImageFont
        import textwrap

        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        h, w = image.shape[:2]

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
        text_y = (h - total_text_height) // 2

        for line in lines:
            text_length = draw.textlength(line, font=font)
            text_x = (w - text_length) // 2

            # Draw text with outline for OSB (better visibility)
            outline_color = (255, 255, 255)
            text_color = (0, 0, 0)

            # Draw outline (4 directions)
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((text_x + dx, text_y + dy), line, font=font, fill=outline_color)

            # Draw main text
            draw.text((text_x, text_y), line, font=font, fill=text_color)
            text_y += line_height

        # Convert back
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

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
    # NEW: Combined Architecture options
    mode: str = "realtime",
    detect_osb: bool = True,
    inpaint_method: str = "auto",
) -> Tuple[Image.Image, dict]:
    """
    Convenience function to translate a manga image.

    Args:
        image_path: Path to input image
        output_path: Optional path to save result
        source_lang: Source language code
        target_lang: Target language code
        font_path: Path to font file
        mode: Processing mode ("realtime", "quality", "premium")
        detect_osb: Whether to detect text outside speech bubbles
        inpaint_method: Inpainting method ("auto", "canvas", "opencv", "flux")

    Returns:
        Tuple of (translated PIL Image, timing dict)
    """
    pipeline = MangaPipeline(
        source_lang=source_lang,
        target_lang=target_lang,
        font_path=font_path,
        mode=mode,
        detect_osb=detect_osb,
        inpaint_method=inpaint_method,
    )

    image = Image.open(image_path)
    result, timing = pipeline.translate_image(image)

    if output_path:
        result.save(output_path)
        logger.info(f"Saved to: {output_path}")

    return result, timing


if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Manga Translation Pipeline")
    parser.add_argument("image_path", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path")
    parser.add_argument("-m", "--mode", default="realtime",
                       choices=["realtime", "quality", "premium"],
                       help="Processing mode")
    parser.add_argument("--source-lang", default="ja", help="Source language")
    parser.add_argument("--target-lang", default="id", help="Target language")
    parser.add_argument("--no-osb", action="store_true", help="Disable OSB detection")
    parser.add_argument("--inpaint", default="auto",
                       choices=["auto", "canvas", "opencv", "flux"],
                       help="Inpainting method")

    args = parser.parse_args()

    # Preload models for the selected mode
    print(f"Loading models for {args.mode} mode...")
    Services.preload_for_mode(args.mode, args.source_lang)

    # Translate
    print(f"Processing: {args.image_path}")
    result, timing = translate_manga(
        args.image_path,
        args.output,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        mode=args.mode,
        detect_osb=not args.no_osb,
        inpaint_method=args.inpaint,
    )

    # Print timing info
    print("\nTiming Information:")
    for key, value in timing.items():
        if key.endswith("_ms"):
            print(f"  {key}: {value}ms")
        else:
            print(f"  {key}: {value}")

    if not args.output:
        result.show()
