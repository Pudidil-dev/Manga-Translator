"""
Singleton services for optimized model loading.
Models are loaded once and reused across all requests.
Auto-detects GPU: NVIDIA CUDA or Intel GPU (via OpenVINO).
"""

import os
import logging
from threading import Lock
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Auto-detect GPU type
USE_CUDA = torch.cuda.is_available()
CUDA_NAME = torch.cuda.get_device_name(0) if USE_CUDA else None

# Detect Intel GPU via OpenVINO
USE_INTEL_GPU = False
INTEL_GPU_NAME = None
try:
    from openvino import Core
    core = Core()
    available_devices = core.available_devices
    if "GPU" in available_devices:
        USE_INTEL_GPU = True
        # Get Intel GPU name
        try:
            INTEL_GPU_NAME = core.get_property("GPU", "FULL_DEVICE_NAME")
        except Exception:
            INTEL_GPU_NAME = "Intel GPU"
        logger.info(f"Intel GPU detected via OpenVINO: {INTEL_GPU_NAME}")
except ImportError:
    pass
except Exception as e:
    logger.debug(f"OpenVINO GPU detection failed: {e}")

if USE_CUDA:
    logger.info(f"NVIDIA CUDA GPU detected: {CUDA_NAME}")
elif USE_INTEL_GPU:
    logger.info(f"Using Intel GPU: {INTEL_GPU_NAME}")
else:
    logger.info("No GPU detected, using CPU optimization")
    # CPU Optimization settings (only when no GPU)
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")


class TransformersSAMPredictor:
    """
    Wrapper for transformers SAM model to mimic SAM2ImagePredictor API.
    Used as fallback when sam2 package is not available.
    """

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor
        self._current_image: Optional[Any] = None
        self._current_embedding: Optional[Any] = None

    def set_image(self, image: np.ndarray) -> None:
        """Set the image for prediction."""
        from PIL import Image as PILImage
        if isinstance(image, np.ndarray):
            self._current_image = PILImage.fromarray(image)
        else:
            self._current_image = image
        self._current_embedding = None  # Clear embedding cache

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict mask for the given prompts.

        Args:
            point_coords: Point coordinates (N, 2)
            point_labels: Point labels (N,)
            box: Box coordinates [x1, y1, x2, y2]
            multimask_output: Whether to return multiple masks

        Returns:
            (masks, scores, logits)
        """
        if self._current_image is None:
            raise ValueError("No image set. Call set_image() first.")

        # Prepare inputs
        if box is not None:
            # Convert box to format expected by processor
            input_boxes = [[box.tolist() if hasattr(box, 'tolist') else list(box)]]
            inputs = self.processor(
                self._current_image,
                input_boxes=input_boxes,
                return_tensors="pt"
            )
        elif point_coords is not None:
            inputs = self.processor(
                self._current_image,
                input_points=[[point_coords.tolist()]],
                input_labels=[[point_labels.tolist()]],
                return_tensors="pt"
            )
        else:
            raise ValueError("Either box or point_coords must be provided")

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process masks
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]

        # Convert to numpy
        masks_np = masks.squeeze(0).numpy() > 0.5
        scores_np = outputs.iou_scores.squeeze(0).cpu().numpy()
        logits_np = outputs.pred_masks.squeeze(0).cpu().numpy()

        if not multimask_output:
            # Return best mask
            best_idx = scores_np.argmax()
            return masks_np[best_idx:best_idx+1], scores_np[best_idx:best_idx+1], logits_np[best_idx:best_idx+1]

        return masks_np, scores_np, logits_np


class Services:
    """
    Singleton service container for ML models.
    Thread-safe lazy loading of YOLO, OCR (hybrid), Translator, and new models.

    New models (Combined Architecture):
    - OSB YOLO: Outside Speech Bubble text detection
    - Conjoined YOLO: Multi-bubble region detection
    - SAM: Segment Anything Model for precise masks
    - Flux: AI inpainting for premium mode
    """

    _yolo = None
    _ocr = None
    _hybrid_ocr = None
    _translator = None
    _yolo_lock = Lock()
    _ocr_lock = Lock()
    _hybrid_ocr_lock = Lock()
    _translator_lock = Lock()

    # NEW: Additional model singletons
    _osb_yolo = None
    _conjoined_yolo = None
    _panel_yolo = None
    _sam = None
    _flux_inpainter = None
    _osb_yolo_lock = Lock()
    _conjoined_yolo_lock = Lock()
    _panel_yolo_lock = Lock()
    _sam_lock = Lock()
    _flux_lock = Lock()

    # Base directory for models
    MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

    # Model paths - Updated with HuggingFace models (ported from MangaTranslator)
    # Primary speech bubble detector
    YOLO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.pt")
    YOLO_MODEL_HF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "yolov8m_seg-speech-bubble.pt")
    YOLO_OPENVINO_FP32 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model_openvino_model")
    YOLO_OPENVINO_INT8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model_int8_openvino_model")
    OCR_WORKERS = int(os.getenv("OCR_WORKERS", "4"))

    # Advanced detection models (from MangaTranslator/HuggingFace)
    OSB_YOLO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "animetext_yolov12x.pt")
    CONJOINED_YOLO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "comic-speech-bubble-detector-yolov8m.pt")
    PANEL_YOLO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "manga109_panel_yolov11.pt")
    SAM_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "sam")

    # Track which model format is loaded
    _yolo_format = None

    @classmethod
    def get_yolo(cls) -> Optional[Any]:
        """
        Get YOLO model instance (singleton).
        Priority: HuggingFace model > Legacy model
        GPU Priority: CUDA GPU > Intel GPU (OpenVINO) > OpenVINO CPU > PyTorch CPU.
        """
        if cls._yolo is None:
            with cls._yolo_lock:
                if cls._yolo is None:
                    from ultralytics import YOLO

                    # Determine which model file to use
                    # Priority: HuggingFace model > Legacy model
                    if os.path.exists(cls.YOLO_MODEL_HF):
                        model_path = cls.YOLO_MODEL_HF
                        logger.info("Using HuggingFace speech bubble model")
                    elif os.path.exists(cls.YOLO_MODEL):
                        model_path = cls.YOLO_MODEL
                        logger.info("Using legacy YOLO model")
                    else:
                        logger.warning("No YOLO model found! Please run: python download_models.py realtime")
                        return None

                    # Priority: CUDA > Intel GPU > OpenVINO CPU > PyTorch
                    if USE_CUDA:
                        # Use PyTorch with NVIDIA CUDA
                        logger.info(f"Loading YOLO with NVIDIA CUDA ({CUDA_NAME})")
                        cls._yolo = YOLO(model_path)
                        cls._yolo.to('cuda')
                        cls._yolo_format = "pytorch_cuda"
                    elif USE_INTEL_GPU and os.path.exists(cls.YOLO_OPENVINO_INT8):
                        # Use OpenVINO INT8 model (CPU optimized, Intel GPU not directly supported by PyTorch)
                        logger.info(f"Loading YOLO with OpenVINO INT8 (Intel optimized)")
                        cls._yolo = YOLO(cls.YOLO_OPENVINO_INT8)
                        cls._yolo_format = "openvino_int8"
                    elif USE_INTEL_GPU and os.path.exists(cls.YOLO_OPENVINO_FP32):
                        logger.info(f"Loading YOLO with OpenVINO FP32 (Intel optimized)")
                        cls._yolo = YOLO(cls.YOLO_OPENVINO_FP32)
                        cls._yolo_format = "openvino_fp32"
                    elif USE_INTEL_GPU:
                        # Intel GPU detected but no OpenVINO model - use PyTorch CPU
                        # Note: PyTorch doesn't support Intel GPU directly, use CPU
                        logger.info(f"Intel GPU detected but using CPU (export to OpenVINO for GPU acceleration)")
                        cls._yolo = YOLO(model_path)
                        cls._yolo_format = "pytorch_cpu"
                    elif os.path.exists(cls.YOLO_OPENVINO_INT8):
                        logger.info("Loading YOLO with OpenVINO INT8 (CPU)")
                        cls._yolo = YOLO(cls.YOLO_OPENVINO_INT8)
                        cls._yolo_format = "openvino_int8_cpu"
                    elif os.path.exists(cls.YOLO_OPENVINO_FP32):
                        logger.info("Loading YOLO with OpenVINO FP32 (CPU)")
                        cls._yolo = YOLO(cls.YOLO_OPENVINO_FP32)
                        cls._yolo_format = "openvino_fp32_cpu"
                    else:
                        logger.info("Loading YOLO PyTorch model (CPU)")
                        logger.info("  Tip: Run 'python export_openvino.py --int8' for speedup")
                        cls._yolo = YOLO(model_path)
                        cls._yolo_format = "pytorch_cpu"

        return cls._yolo

    @classmethod
    def get_yolo_format(cls) -> str:
        """Get the format of loaded YOLO model."""
        if cls._yolo is None:
            cls.get_yolo()
        return cls._yolo_format or "unknown"

    @classmethod
    def get_ocr(cls) -> Optional[Any]:
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
    def get_hybrid_ocr(cls) -> Optional[Any]:
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
    def get_translator(cls, **kwargs: Any) -> Optional[Any]:
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
    def export_yolo_openvino(cls, int8: bool = False) -> str:
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
    def preload_all(cls, source_lang: str = "ja") -> None:
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
    def reset(cls) -> None:
        """
        Reset all singleton instances (for testing).
        """
        cls._yolo = None
        cls._ocr = None
        cls._hybrid_ocr = None
        cls._translator = None
        cls._osb_yolo = None
        cls._conjoined_yolo = None
        cls._sam = None
        cls._flux_inpainter = None

    @classmethod
    def unload_flux(cls) -> None:
        """
        Unload Flux model to free GPU/CPU memory.
        Call this after batch processing to release ~4-6GB memory.
        """
        if cls._flux_inpainter is not None:
            try:
                cls._flux_inpainter.unload()
            except Exception as e:
                logger.warning(f"Error unloading Flux: {e}")
            cls._flux_inpainter = None
            logger.info("Flux inpainter unloaded")

    # =========================================================================
    # NEW: OSB (Outside Speech Bubble) Detection
    # =========================================================================

    @classmethod
    def get_osb_yolo(cls) -> Optional[Any]:
        """
        Get OSB YOLO model instance (singleton).
        Detects text outside speech bubbles: SFX, narration, onomatopoeia.

        Returns:
            YOLO model or None if model file not found
        """
        if cls._osb_yolo is None:
            with cls._osb_yolo_lock:
                if cls._osb_yolo is None:
                    if not os.path.exists(cls.OSB_YOLO_MODEL):
                        logger.warning(
                            f"OSB YOLO model not found at {cls.OSB_YOLO_MODEL}. "
                            "OSB detection will be disabled. "
                            "Download the model to enable this feature."
                        )
                        return None

                    from ultralytics import YOLO

                    logger.info("Loading OSB YOLO model")
                    try:
                        cls._osb_yolo = YOLO(cls.OSB_YOLO_MODEL)
                    except AttributeError as e:
                        # Handle model compatibility issue with fuse()
                        if "has no attribute 'bn'" in str(e):
                            logger.warning("Model fuse() compatibility issue, loading without fuse")
                            cls._osb_yolo = YOLO(cls.OSB_YOLO_MODEL)
                            # Disable fuse to prevent errors
                            if hasattr(cls._osb_yolo, 'model') and hasattr(cls._osb_yolo.model, 'fuse'):
                                cls._osb_yolo.model.fuse = lambda verbose=False: cls._osb_yolo.model
                        else:
                            raise

                    # Use GPU if available
                    if USE_CUDA:
                        cls._osb_yolo.to('cuda')
                        logger.info("OSB YOLO loaded with CUDA")
                    elif USE_INTEL_GPU:
                        # Note: PyTorch doesn't support Intel GPU directly, use CPU
                        logger.info("OSB YOLO loaded (CPU - export to OpenVINO for Intel GPU)")

        return cls._osb_yolo

    @classmethod
    def get_conjoined_yolo(cls) -> Optional[Any]:
        """
        Get Conjoined YOLO model instance (singleton).
        Detects multi-bubble regions (merged speech bubbles).

        Returns:
            YOLO model or None if model file not found
        """
        if cls._conjoined_yolo is None:
            with cls._conjoined_yolo_lock:
                if cls._conjoined_yolo is None:
                    if not os.path.exists(cls.CONJOINED_YOLO_MODEL):
                        logger.warning(
                            f"Conjoined YOLO model not found at {cls.CONJOINED_YOLO_MODEL}. "
                            "Conjoined detection will be disabled."
                        )
                        return None

                    from ultralytics import YOLO

                    logger.info("Loading Conjoined YOLO model")
                    cls._conjoined_yolo = YOLO(cls.CONJOINED_YOLO_MODEL)

                    if USE_CUDA:
                        cls._conjoined_yolo.to('cuda')
                        logger.info("Conjoined YOLO loaded with CUDA")
                    elif USE_INTEL_GPU:
                        # Note: PyTorch doesn't support Intel GPU directly, use CPU
                        logger.info("Conjoined YOLO loaded (CPU - export to OpenVINO for Intel GPU)")

        return cls._conjoined_yolo

    # =========================================================================
    # NEW: SAM (Segment Anything Model) for precise segmentation
    # =========================================================================

    @classmethod
    def get_sam(cls, version: str = "sam2") -> Optional[Any]:
        """
        Get SAM model instance (singleton).
        Used for precise bubble mask segmentation in quality/premium modes.

        Args:
            version: SAM version ("sam2" or "sam3")

        Returns:
            SAM predictor or None if not available
        """
        if cls._sam is None:
            with cls._sam_lock:
                if cls._sam is None:
                    try:
                        if version == "sam2":
                            # Try SAM2 from sam2 package
                            try:
                                from sam2.sam2_image_predictor import SAM2ImagePredictor
                                logger.info("Loading SAM2 model from HuggingFace")
                                cls._sam = SAM2ImagePredictor.from_pretrained(
                                    "facebook/sam2.1-hiera-large"
                                )
                                logger.info("SAM2 model loaded successfully")
                            except ImportError:
                                # Fallback to transformers SAM
                                try:
                                    from transformers import SamModel, SamProcessor
                                    logger.info("Loading SAM from transformers")
                                    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
                                    model = SamModel.from_pretrained("facebook/sam-vit-base")
                                    if USE_CUDA:
                                        model = model.to("cuda")
                                    # Create a wrapper that mimics SAM2ImagePredictor API
                                    cls._sam = TransformersSAMPredictor(model, processor)
                                    logger.info("SAM (transformers) loaded successfully")
                                except ImportError:
                                    logger.warning(
                                        "SAM2 and transformers SAM not installed. "
                                        "Install with: pip install sam2 or pip install transformers"
                                    )
                                    return None
                        else:
                            logger.warning(f"SAM version {version} not supported")
                            return None

                    except Exception as e:
                        logger.error(f"Failed to load SAM: {e}")
                        return None

        return cls._sam

    # =========================================================================
    # NEW: Flux Inpainting for premium mode
    # =========================================================================

    @classmethod
    def get_flux_inpainter(cls) -> Optional[Any]:
        """
        Get Flux inpainting model (singleton).
        Used for AI-powered text removal in premium mode.

        Returns:
            FluxInpainter instance or None if not available
        """
        if cls._flux_inpainter is None:
            with cls._flux_lock:
                if cls._flux_inpainter is None:
                    try:
                        from inpainting import FluxInpainter

                        logger.info("Initializing Flux inpainter...")
                        cls._flux_inpainter = FluxInpainter(
                            model_id="black-forest-labs/FLUX.1-Fill-dev",
                            device="cuda" if USE_CUDA else "cpu"
                        )

                        # Try to preload the model
                        if cls._flux_inpainter.load():
                            logger.info("Flux inpainter ready")
                        else:
                            logger.warning("Flux model could not be loaded (will retry on use)")

                    except ImportError as e:
                        logger.warning(
                            f"Flux dependencies not installed: {e}. "
                            "Install with: pip install diffusers transformers accelerate"
                        )
                        return None
                    except Exception as e:
                        logger.error(f"Failed to initialize Flux: {e}")
                        return None

        return cls._flux_inpainter

    # =========================================================================
    # Mode-based preloading
    # =========================================================================

    @classmethod
    def preload_for_mode(cls, mode: str = "realtime", source_lang: str = "ja") -> None:
        """
        Preload models based on processing mode.

        Args:
            mode: Processing mode (realtime, quality, premium)
            source_lang: Source language for OCR preloading
        """
        logger.info(f"Preloading models for {mode} mode...")

        # Always load: bubble YOLO, hybrid OCR, translator
        cls.get_yolo()
        hybrid_ocr = cls.get_hybrid_ocr()
        hybrid_ocr.preload(source_lang)
        cls.get_translator()

        if mode in ("realtime", "quality", "premium"):
            # Load OSB detector for all modes
            cls.get_osb_yolo()

        if mode in ("quality", "premium"):
            # Load conjoined detector
            cls.get_conjoined_yolo()
            # Load SAM for precise masks
            cls.get_sam()

        if mode == "premium":
            # Load Flux inpainter (optional, may fail)
            cls.get_flux_inpainter()

        logger.info(f"Models preloaded for {mode} mode")

    @classmethod
    def get_loaded_models(cls) -> Dict[str, str]:
        """
        Get status of all loaded models.

        Returns:
            Dict with model names and their status (loaded/unloaded)
        """
        return {
            "bubble_yolo": "loaded" if cls._yolo else "unloaded",
            "osb_yolo": "loaded" if cls._osb_yolo else "unloaded",
            "conjoined_yolo": "loaded" if cls._conjoined_yolo else "unloaded",
            "manga_ocr": "loaded" if cls._ocr else "unloaded",
            "hybrid_ocr": "loaded" if cls._hybrid_ocr else "unloaded",
            "translator": "loaded" if cls._translator else "unloaded",
            "sam": "loaded" if cls._sam else "unloaded",
            "flux": "loaded" if cls._flux_inpainter else "unloaded",
        }

    @classmethod
    def get_gpu_info(cls) -> Dict[str, Optional[Union[bool, str]]]:
        """Get GPU information."""
        return {
            "cuda_available": USE_CUDA,
            "cuda_name": CUDA_NAME,
            "intel_gpu_available": USE_INTEL_GPU,
            "intel_gpu_name": INTEL_GPU_NAME,
        }


# Export commonly used constants
OCR_WORKERS = Services.OCR_WORKERS
