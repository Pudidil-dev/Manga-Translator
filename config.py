"""
Processing Configuration for Manga-Translator.
Dataclass-based configuration system (inspired by MangaTranslator).
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import os


@dataclass
class DetectionConfig:
    """Configuration for text detection."""
    bubble_confidence: float = 0.5
    osb_confidence: float = 0.5
    conjoined_confidence: float = 0.35
    enable_osb: bool = True
    enable_conjoined: bool = True


@dataclass
class InpaintingConfig:
    """Configuration for text inpainting/removal."""
    method: Literal["auto", "canvas", "opencv", "flux"] = "auto"
    opencv_algorithm: Literal["telea", "ns"] = "telea"
    flux_steps: int = 4


@dataclass
class TranslationConfig:
    """Configuration for translation."""
    source_lang: str = "ja"
    target_lang: str = "id"
    provider: str = "gemini"

    # API configuration from environment
    gemini_api_url: str = field(default_factory=lambda: os.getenv("GEMINI_API_URL", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))


@dataclass
class RenderingConfig:
    """Configuration for text rendering."""
    font_path: str = "fonts/animeace_i.ttf"
    max_font_size: int = 16
    min_font_size: int = 8
    line_height: int = 16
    wrapping_ratio: float = 0.075


@dataclass
class ProcessingConfig:
    """
    Main processing configuration.
    Supports 3 modes: realtime, quality, premium.
    """
    mode: Literal["realtime", "quality", "premium"] = "realtime"
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    inpainting: InpaintingConfig = field(default_factory=InpaintingConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)

    # OCR settings
    ocr_workers: int = field(default_factory=lambda: int(os.getenv("OCR_WORKERS", "4")))

    # SAM segmentation (quality/premium only)
    use_sam: bool = False
    sam_model: Literal["sam2", "sam3"] = "sam2"

    @classmethod
    def for_realtime(cls) -> "ProcessingConfig":
        """
        Realtime mode configuration.
        Target: <1 second processing time.
        - Basic detection (bubble + OSB)
        - Canvas overlay inpainting
        - No SAM segmentation
        """
        return cls(
            mode="realtime",
            detection=DetectionConfig(
                enable_osb=True,
                enable_conjoined=False,
            ),
            inpainting=InpaintingConfig(method="canvas"),
            use_sam=False,
        )

    @classmethod
    def for_quality(cls) -> "ProcessingConfig":
        """
        Quality mode configuration.
        Target: 2-3 seconds processing time.
        - Full detection (bubble + OSB + conjoined)
        - OpenCV inpainting
        - SAM2 segmentation for precise masks
        """
        return cls(
            mode="quality",
            detection=DetectionConfig(
                enable_osb=True,
                enable_conjoined=True,
            ),
            inpainting=InpaintingConfig(method="opencv"),
            use_sam=True,
            sam_model="sam2",
        )

    @classmethod
    def for_premium(cls) -> "ProcessingConfig":
        """
        Premium mode configuration.
        Target: 5-10 seconds processing time.
        - Full detection
        - Flux AI inpainting
        - SAM2/SAM3 segmentation
        """
        return cls(
            mode="premium",
            detection=DetectionConfig(
                enable_osb=True,
                enable_conjoined=True,
            ),
            inpainting=InpaintingConfig(method="flux", flux_steps=4),
            use_sam=True,
            sam_model="sam2",
        )

    @classmethod
    def from_mode(cls, mode: str) -> "ProcessingConfig":
        """Create configuration from mode string."""
        if mode == "quality":
            return cls.for_quality()
        elif mode == "premium":
            return cls.for_premium()
        else:
            return cls.for_realtime()


# Language mappings
LANGUAGE_MAP = {
    "english": "en",
    "indonesian": "id",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "portuguese": "pt",
    "russian": "ru",
    "arabic": "ar",
    "thai": "th",
    "vietnamese": "vi",
    "chinese (simplified)": "zh",
    "chinese (traditional)": "zh-tw",
    "korean": "ko",
    "japanese": "ja",
    "malay": "ms",
    "hindi": "hi",
    "italian": "it",
    "dutch": "nl",
    "polish": "pl",
    "turkish": "tr",
}

SOURCE_LANGUAGE_MAP = {
    "japanese": "ja",
    "korean": "ko",
    "chinese (simplified)": "zh",
    "chinese (traditional)": "zh-tw",
    "english": "en",
}
