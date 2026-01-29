"""
Singleton services for optimized model loading.
Models are loaded once and reused across all requests.
Auto-detects CUDA GPU for acceleration.
"""

import os
import logging
from threading import Lock
from typing import Optional
import torch

logger = logging.getLogger(__name__)

# Auto-detect GPU
USE_GPU = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if USE_GPU else None

if USE_GPU:
    logger.info(f"CUDA GPU detected: {GPU_NAME}")
else:
    logger.info("No CUDA GPU, using CPU optimization")
    # CPU Optimization settings (only when no GPU)
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")


class Services:
    """
    Singleton service container for ML models.
    Thread-safe lazy loading of YOLO, OCR (hybrid), and Translator.
    """

    _yolo = None
    _ocr = None
    _hybrid_ocr = None
    _translator = None
    _yolo_lock = Lock()
    _ocr_lock = Lock()
    _hybrid_ocr_lock = Lock()
    _translator_lock = Lock()

    # Configuration
    MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
    YOLO_MODEL = os.path.join(MODEL_DIR, "model", "model.pt")
    YOLO_OPENVINO_FP32 = os.path.join(MODEL_DIR, "model", "model_openvino_model")
    YOLO_OPENVINO_INT8 = os.path.join(MODEL_DIR, "model", "model_int8_openvino_model")
    OCR_WORKERS = int(os.getenv("OCR_WORKERS", "4"))

    # Track which model format is loaded
    _yolo_format = None

    @classmethod
    def get_yolo(cls):
        """
        Get YOLO model instance (singleton).
        Priority: CUDA GPU > OpenVINO INT8 > OpenVINO FP32 > PyTorch CPU.
        """
        if cls._yolo is None:
            with cls._yolo_lock:
                if cls._yolo is None:
                    from ultralytics import YOLO

                    # Priority: CUDA > OpenVINO INT8 > FP32 > PyTorch
                    if USE_GPU:
                        # Use PyTorch with CUDA
                        logger.info(f"Loading YOLO with CUDA GPU ({GPU_NAME})")
                        cls._yolo = YOLO(cls.YOLO_MODEL)
                        cls._yolo.to('cuda')
                        cls._yolo_format = "pytorch_cuda"
                    elif os.path.exists(cls.YOLO_OPENVINO_INT8):
                        logger.info("Loading YOLO with OpenVINO INT8 (fastest CPU)")
                        cls._yolo = YOLO(cls.YOLO_OPENVINO_INT8)
                        cls._yolo_format = "openvino_int8"
                    elif os.path.exists(cls.YOLO_OPENVINO_FP32):
                        logger.info("Loading YOLO with OpenVINO FP32")
                        cls._yolo = YOLO(cls.YOLO_OPENVINO_FP32)
                        cls._yolo_format = "openvino_fp32"
                    else:
                        logger.info("Loading YOLO PyTorch model (CPU)")
                        logger.info("  Tip: Run 'python export_openvino.py --int8' for 2-3x speedup")
                        cls._yolo = YOLO(cls.YOLO_MODEL)
                        cls._yolo_format = "pytorch_cpu"

        return cls._yolo

    @classmethod
    def get_yolo_format(cls) -> str:
        """Get the format of loaded YOLO model."""
        if cls._yolo is None:
            cls.get_yolo()
        return cls._yolo_format or "unknown"

    @classmethod
    def get_ocr(cls):
        """
        Get MangaOCR instance (singleton).
        For Japanese only. Use get_hybrid_ocr() for multi-language.
        """
        if cls._ocr is None:
            with cls._ocr_lock:
                if cls._ocr is None:
                    from manga_ocr import MangaOcr

                    logger.info("Loading MangaOCR model")
                    cls._ocr = MangaOcr()

        return cls._ocr

    @classmethod
    def get_hybrid_ocr(cls):
        """
        Get HybridOCR instance (singleton).
        Supports multiple languages: ja, ko, zh, zh-tw, en.
        - Japanese: uses MangaOCR
        - Others: uses EasyOCR
        """
        if cls._hybrid_ocr is None:
            with cls._hybrid_ocr_lock:
                if cls._hybrid_ocr is None:
                    from ocr_hybrid import HybridOCR

                    logger.info("Initializing HybridOCR")
                    cls._hybrid_ocr = HybridOCR()

        return cls._hybrid_ocr

    @classmethod
    def get_translator(cls, **kwargs):
        """
        Get MangaTranslator instance (singleton).
        """
        if cls._translator is None:
            with cls._translator_lock:
                if cls._translator is None:
                    from translator.translator import MangaTranslator

                    logger.info("Initializing MangaTranslator")
                    cls._translator = MangaTranslator(**kwargs)

        return cls._translator

    @classmethod
    def export_yolo_openvino(cls, int8: bool = False):
        """
        Export YOLO model to OpenVINO format for Intel CPU optimization.
        Run this once to create optimized model.

        Args:
            int8: If True, export INT8 quantized model (faster, slightly less accurate)
        """
        from ultralytics import YOLO

        target_path = cls.YOLO_OPENVINO_INT8 if int8 else cls.YOLO_OPENVINO_FP32
        format_name = "INT8" if int8 else "FP32"

        if os.path.exists(target_path):
            logger.info(f"OpenVINO {format_name} model already exists")
            return target_path

        logger.info(f"Exporting YOLO to OpenVINO {format_name} format...")
        model = YOLO(cls.YOLO_MODEL)

        if int8:
            export_path = model.export(format="openvino", int8=True)
        else:
            export_path = model.export(format="openvino", half=False)

        logger.info(f"Exported to: {export_path}")
        return export_path

    @classmethod
    def preload_all(cls, source_lang: str = "ja"):
        """
        Preload all models at startup.
        Call this during app initialization to avoid cold start.

        Args:
            source_lang: Source language to preload OCR for (ja, ko, zh, en)
        """
        logger.info("Preloading all models...")
        cls.get_yolo()

        # Preload hybrid OCR for the specified language
        hybrid_ocr = cls.get_hybrid_ocr()
        hybrid_ocr.preload(source_lang)

        cls.get_translator()
        logger.info("All models loaded")

    @classmethod
    def reset(cls):
        """
        Reset all singleton instances (for testing).
        """
        cls._yolo = None
        cls._ocr = None
        cls._hybrid_ocr = None
        cls._translator = None


# Export commonly used constants
OCR_WORKERS = Services.OCR_WORKERS
