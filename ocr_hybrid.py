"""
Hybrid Multi-Language OCR Service.
Uses MangaOCR for Japanese, EasyOCR for other languages.
Auto-detects CUDA GPU for acceleration.
Note: EasyOCR only supports NVIDIA CUDA, not Intel GPU.
"""

import logging
import os
from threading import Lock
from typing import Union
from PIL import Image
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Auto-detect CUDA GPU (EasyOCR only supports CUDA, not Intel GPU)
USE_CUDA_FOR_OCR = torch.cuda.is_available()
if USE_CUDA_FOR_OCR:
    logger.info(f"OCR: CUDA GPU detected: {torch.cuda.get_device_name(0)}")
else:
    logger.info("OCR: No CUDA GPU, EasyOCR will use CPU")

# Supported languages
SUPPORTED_LANGUAGES = {
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "en": "English",
}

# OCR engine mapping
OCR_ENGINE_MAP = {
    "ja": "manga_ocr",      # Best for Japanese manga
    "ko": "easyocr",        # Korean
    "zh": "easyocr",        # Chinese Simplified
    "zh-tw": "easyocr",     # Chinese Traditional
    "en": "easyocr",        # English
}

# EasyOCR language codes
EASYOCR_LANG_MAP = {
    "ko": ["ko"],
    "zh": ["ch_sim"],
    "zh-tw": ["ch_tra"],
    "en": ["en"],
    "ja": ["ja"],  # Fallback if MangaOCR not available
}


class HybridOCR:
    """
    Hybrid OCR that uses the best engine for each language.
    - Japanese: MangaOCR (specialized for manga)
    - Korean/Chinese/English: EasyOCR (multi-language, stable)
    """

    def __init__(self):
        self._manga_ocr = None
        self._easyocr_readers = {}
        self._easyocr_available = None
        self._manga_ocr_lock = Lock()
        self._easyocr_lock = Lock()  # Prevent concurrent downloads

    def _check_easyocr_available(self) -> bool:
        """Check if EasyOCR is installed."""
        if self._easyocr_available is None:
            try:
                import easyocr
                self._easyocr_available = True
            except ImportError:
                logger.warning(
                    "EasyOCR not installed. Install with: pip install easyocr"
                )
                self._easyocr_available = False
        return self._easyocr_available

    def _get_manga_ocr(self):
        """Get MangaOCR instance (lazy loading, thread-safe)."""
        if self._manga_ocr is None:
            with self._manga_ocr_lock:
                if self._manga_ocr is None:
                    logger.info("Loading MangaOCR...")
                    from manga_ocr import MangaOcr
                    self._manga_ocr = MangaOcr()
                    logger.info("MangaOCR loaded")
        return self._manga_ocr

    def _get_easyocr_reader(self, lang: str):
        """Get EasyOCR reader for specific language (lazy loading, thread-safe)."""
        lang_codes = EASYOCR_LANG_MAP.get(lang, ["en"])
        lang_key = "_".join(lang_codes)

        if lang_key not in self._easyocr_readers:
            # Use lock to prevent concurrent downloads (causes Windows file lock issues)
            with self._easyocr_lock:
                # Double-check after acquiring lock
                if lang_key not in self._easyocr_readers:
                    if not self._check_easyocr_available():
                        raise RuntimeError("EasyOCR not installed")

                    logger.info(f"Loading EasyOCR for {lang} ({lang_codes})...")
                    import easyocr

                    self._easyocr_readers[lang_key] = easyocr.Reader(
                        lang_codes,
                        gpu=USE_CUDA_FOR_OCR,
                        verbose=False,
                    )
                    logger.info(f"EasyOCR ({lang}) loaded")

        return self._easyocr_readers[lang_key]

    def read(self, image: Union[Image.Image, np.ndarray], lang: str = "ja") -> str:
        """
        Read text from image using appropriate OCR engine.

        Args:
            image: PIL Image or numpy array
            lang: Source language code (ja, ko, zh, zh-tw, en)

        Returns:
            Extracted text string
        """
        engine = OCR_ENGINE_MAP.get(lang, "easyocr")

        if engine == "manga_ocr":
            return self._read_manga_ocr(image)
        else:
            return self._read_easyocr(image, lang)

    def _read_manga_ocr(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Read with MangaOCR (Japanese)."""
        ocr = self._get_manga_ocr()

        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        try:
            text = ocr(image)
            return text if text else ""
        except Exception as e:
            logger.warning(f"MangaOCR failed: {e}")
            return ""

    def _read_easyocr(self, image: Union[Image.Image, np.ndarray], lang: str) -> str:
        """Read with EasyOCR (Korean/Chinese/English)."""
        reader = self._get_easyocr_reader(lang)

        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        try:
            # EasyOCR returns list of (bbox, text, confidence)
            results = reader.readtext(image)

            if not results:
                return ""

            # Extract text from results
            texts = [result[1] for result in results if result[1]]
            return " ".join(texts)

        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
            return ""

    def get_engine_name(self, lang: str) -> str:
        """Get the OCR engine name for a language."""
        engine = OCR_ENGINE_MAP.get(lang, "easyocr")
        if engine == "manga_ocr":
            return "MangaOCR"
        lang_codes = EASYOCR_LANG_MAP.get(lang, ["en"])
        return f"EasyOCR ({', '.join(lang_codes)})"

    def preload(self, lang: str = "ja"):
        """Preload OCR engine for specific language only."""
        engine = OCR_ENGINE_MAP.get(lang, "easyocr")
        if engine == "manga_ocr":
            self._get_manga_ocr()
        # Don't preload EasyOCR - it will download models
        # Let it load lazily on first use

    def preload_all(self):
        """
        Preload essential OCR engines only.
        EasyOCR models are loaded lazily to avoid multiple downloads.
        """
        logger.info("Preloading essential OCR engines...")

        # Only load MangaOCR for Japanese (most common use case)
        try:
            self._get_manga_ocr()
        except Exception as e:
            logger.warning(f"Failed to load MangaOCR: {e}")

        # EasyOCR will be loaded on-demand to avoid downloading
        # multiple models at startup
        logger.info("OCR engines loaded (EasyOCR will load on first use)")


# Singleton instance
_hybrid_ocr_instance = None


def get_hybrid_ocr() -> HybridOCR:
    """Get singleton HybridOCR instance."""
    global _hybrid_ocr_instance
    if _hybrid_ocr_instance is None:
        _hybrid_ocr_instance = HybridOCR()
    return _hybrid_ocr_instance
