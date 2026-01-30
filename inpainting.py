"""
Inpainting module for text removal.
Supports multiple methods: canvas overlay, OpenCV, and Flux AI.

Inspired by MangaTranslator's cleaning and inpainting system.
"""

import logging
from typing import List, Tuple, Optional, Literal
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)


class CanvasOverlay:
    """
    Fast overlay inpainting for realtime mode.
    Simply fills detection regions with detected background color.
    """

    def apply(
        self,
        image: Image.Image,
        regions: List,
        fill_color: Tuple[int, int, int] = None
    ) -> Image.Image:
        """
        Apply canvas overlay to fill text regions.

        Args:
            image: Input PIL Image
            regions: List of regions with bbox attribute
            fill_color: Optional fixed fill color (auto-detect if None)

        Returns:
            Image with text regions filled
        """
        result = image.copy()
        draw = ImageDraw.Draw(result)

        for region in regions:
            # Get bbox (handle both TextRegion and raw tuple)
            if hasattr(region, 'bbox'):
                x1, y1, x2, y2 = region.bbox
            elif len(region) >= 4:
                x1, y1, x2, y2 = int(region[0]), int(region[1]), int(region[2]), int(region[3])
            else:
                continue

            if fill_color:
                color = fill_color
            else:
                # Auto-detect background color
                color = self._detect_background(image, (x1, y1, x2, y2))

            draw.rectangle([x1, y1, x2, y2], fill=color)

        return result

    def _detect_background(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int]:
        """
        Detect dominant background color from border pixels.

        Args:
            image: Input PIL Image
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            RGB color tuple
        """
        x1, y1, x2, y2 = bbox
        img_array = np.array(image)

        # Handle grayscale images
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        h, w = img_array.shape[:2]
        border_pixels = []

        # Sample border pixels (2px around the bbox)
        border_size = 2

        # Top border
        if y1 > border_size:
            top = img_array[max(0, y1-border_size):y1, x1:x2]
            if top.size > 0:
                border_pixels.extend(top.reshape(-1, 3))

        # Bottom border
        if y2 < h - border_size:
            bottom = img_array[y2:min(h, y2+border_size), x1:x2]
            if bottom.size > 0:
                border_pixels.extend(bottom.reshape(-1, 3))

        # Left border
        if x1 > border_size:
            left = img_array[y1:y2, max(0, x1-border_size):x1]
            if left.size > 0:
                border_pixels.extend(left.reshape(-1, 3))

        # Right border
        if x2 < w - border_size:
            right = img_array[y1:y2, x2:min(w, x2+border_size)]
            if right.size > 0:
                border_pixels.extend(right.reshape(-1, 3))

        if not border_pixels:
            return (255, 255, 255)  # Default white

        # Get median color (more robust than mean)
        border_array = np.array(border_pixels)
        median_color = np.median(border_array, axis=0).astype(int)

        return tuple(median_color)


class OpenCVInpainter:
    """
    OpenCV-based inpainting for quality mode.
    Supports TELEA and Navier-Stokes algorithms.
    """

    def __init__(self, algorithm: Literal["telea", "ns"] = "telea"):
        """
        Initialize OpenCV inpainter.

        Args:
            algorithm: "telea" (fast) or "ns" (Navier-Stokes, better quality)
        """
        self.algorithm = algorithm
        self.cv2_flag = (
            cv2.INPAINT_TELEA if algorithm == "telea"
            else cv2.INPAINT_NS
        )

    def apply(
        self,
        image: Image.Image,
        regions: List,
        masks: List[np.ndarray] = None,
        inpaint_radius: int = 3
    ) -> Image.Image:
        """
        Apply OpenCV inpainting to remove text.

        Args:
            image: Input PIL Image
            regions: List of regions with bbox attribute
            masks: Optional list of precise masks (from SAM)
            inpaint_radius: Inpainting radius in pixels

        Returns:
            Inpainted image
        """
        img_array = np.array(image)

        # Convert RGBA to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Create combined mask
        h, w = img_array.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for i, region in enumerate(regions):
            if masks and i < len(masks) and masks[i] is not None:
                # Use SAM mask if available
                mask = masks[i]
                if mask.shape[:2] == (h, w):
                    combined_mask = np.maximum(combined_mask, mask.astype(np.uint8) * 255)
                else:
                    # Resize mask to match image
                    resized_mask = cv2.resize(mask.astype(np.uint8), (w, h))
                    combined_mask = np.maximum(combined_mask, resized_mask * 255)
            else:
                # Use bbox as mask
                if hasattr(region, 'bbox'):
                    x1, y1, x2, y2 = region.bbox
                elif len(region) >= 4:
                    x1, y1, x2, y2 = int(region[0]), int(region[1]), int(region[2]), int(region[3])
                else:
                    continue

                # Clip to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                combined_mask[y1:y2, x1:x2] = 255

        # Apply inpainting
        if np.any(combined_mask > 0):
            inpainted = cv2.inpaint(
                img_array,
                combined_mask,
                inpaintRadius=inpaint_radius,
                flags=self.cv2_flag
            )
        else:
            inpainted = img_array

        return Image.fromarray(inpainted)


class FluxInpainter:
    """
    Flux AI-based inpainting for premium mode.
    Uses FLUX.1-Fill-dev for high-quality text removal.
    """

    def __init__(self, model_id: str = None, device: str = None):
        """
        Initialize Flux inpainter.

        Args:
            model_id: HuggingFace model ID (default: FLUX.1-Fill-dev)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        # Use the official Flux inpainting model
        self.model_id = model_id or "black-forest-labs/FLUX.1-Fill-dev"
        self.device = device
        self.pipeline = None
        self._loaded = False

    def load(self) -> bool:
        """
        Load the Flux model.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._loaded:
            return True

        try:
            import torch
            from diffusers import FluxFillPipeline

            logger.info(f"Loading Flux inpainting model: {self.model_id}")

            # Determine device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Check VRAM - Flux Fill needs ~24GB for full precision
            if self.device == "cuda":
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU VRAM: {vram_gb:.1f}GB")
                if vram_gb < 12:
                    logger.warning("Low VRAM detected. Flux may run slowly or fail.")

            # Load pipeline with optimizations
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.pipeline = FluxFillPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            )

            # Enable memory optimizations
            if self.device == "cuda":
                try:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("Enabled CPU offload for memory optimization")
                except Exception as e:
                    logger.warning(f"Could not enable CPU offload: {e}")
                    self.pipeline.to(self.device)
            else:
                self.pipeline.to(self.device)

            self._loaded = True
            logger.info(f"Flux inpainter loaded on {self.device}")
            return True

        except ImportError as e:
            logger.warning(f"Flux dependencies not installed: {e}")
            logger.warning("Install with: pip install diffusers transformers accelerate")
            return False

        except Exception as e:
            logger.error(f"Failed to load Flux model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def apply(
        self,
        image: Image.Image,
        regions: List,
        masks: List[np.ndarray] = None,
        prompt: str = "clean white background, seamless texture, no text",
        num_inference_steps: int = 28,
        guidance_scale: float = 30.0,
    ) -> Image.Image:
        """
        Apply Flux AI inpainting to remove text.

        Args:
            image: Input PIL Image
            regions: List of regions with bbox attribute
            masks: Optional list of precise masks (from SAM)
            prompt: Prompt for inpainting (what to fill with)
            num_inference_steps: Number of diffusion steps (28 recommended for Fill)
            guidance_scale: Guidance scale (30 recommended for Fill)

        Returns:
            Inpainted image
        """
        if not self._loaded:
            if not self.load():
                logger.warning("Flux model not available")
                return image

        # Create combined mask
        w, h = image.size  # PIL uses (width, height)
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for i, region in enumerate(regions):
            if masks and i < len(masks) and masks[i] is not None:
                mask = masks[i]
                if mask.shape[:2] == (h, w):
                    combined_mask = np.maximum(combined_mask, (mask > 0).astype(np.uint8) * 255)
                else:
                    resized_mask = cv2.resize(mask.astype(np.uint8), (w, h))
                    combined_mask = np.maximum(combined_mask, resized_mask * 255)
            else:
                if hasattr(region, 'bbox'):
                    x1, y1, x2, y2 = region.bbox
                elif len(region) >= 4:
                    x1, y1, x2, y2 = int(region[0]), int(region[1]), int(region[2]), int(region[3])
                else:
                    continue

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Create ellipse mask for speech bubbles
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                cv2.ellipse(combined_mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)

        if not np.any(combined_mask > 0):
            return image

        # Convert mask to PIL
        mask_image = Image.fromarray(combined_mask).convert("L")

        # Dilate mask slightly for better blending
        mask_image = mask_image.filter(ImageFilter.MaxFilter(5))

        try:
            # Resize to multiple of 8 for Flux (required)
            orig_size = image.size
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8

            if new_w != w or new_h != h:
                resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                resized_mask = mask_image.resize((new_w, new_h), Image.Resampling.NEAREST)
            else:
                resized_image = image
                resized_mask = mask_image

            # Run FLUX.1-Fill inpainting
            logger.info(f"Running Flux Fill inpainting on {new_w}x{new_h} image...")
            result = self.pipeline(
                prompt=prompt,
                image=resized_image.convert("RGB"),
                mask_image=resized_mask,
                height=new_h,
                width=new_w,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=512,
            ).images[0]

            # Resize back if needed
            if result.size != orig_size:
                result = result.resize(orig_size, Image.Resampling.LANCZOS)

            logger.info(f"Flux inpainting completed for {len(regions)} regions")
            return result

        except Exception as e:
            logger.error(f"Flux inpainting failed: {e}")
            return image

    def unload(self):
        """Unload model to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self._loaded = False

            # Force garbage collection
            import gc
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Flux inpainter unloaded")


class Inpainter:
    """
    Unified inpainting interface.
    Selects appropriate method based on configuration.
    """

    def __init__(self, method: Literal["auto", "canvas", "opencv", "flux"] = "auto"):
        """
        Initialize inpainter with specified method.

        Args:
            method: Inpainting method
                - "auto": Auto-select based on region analysis
                - "canvas": Fast overlay (realtime)
                - "opencv": OpenCV TELEA/NS (quality)
                - "flux": Flux AI inpainting (premium)
        """
        self.method = method
        self._canvas = CanvasOverlay()
        self._opencv = None
        self._flux = None

    def inpaint(
        self,
        image: Image.Image,
        regions: List,
        masks: List[np.ndarray] = None,
        mode: str = "realtime"
    ) -> Image.Image:
        """
        Inpaint text regions.

        Args:
            image: Input PIL Image
            regions: List of text regions to inpaint
            masks: Optional SAM masks for precise inpainting
            mode: Processing mode (affects auto selection)

        Returns:
            Inpainted image
        """
        if not regions:
            return image

        # Select method
        method = self._select_method(mode, regions)

        logger.info(f"Inpainting {len(regions)} regions with method: {method}")

        if method == "canvas":
            return self._canvas.apply(image, regions)

        elif method == "opencv":
            if self._opencv is None:
                self._opencv = OpenCVInpainter("telea")
            return self._opencv.apply(image, regions, masks)

        elif method == "flux":
            return self._flux_inpaint(image, regions, masks)

        else:
            # Fallback to canvas
            return self._canvas.apply(image, regions)

    def _select_method(self, mode: str, regions: List) -> str:
        """Select inpainting method based on mode and configuration."""
        if self.method != "auto":
            return self.method

        # Auto selection based on mode
        if mode == "realtime":
            return "canvas"
        elif mode == "quality":
            return "opencv"
        elif mode == "premium":
            return "flux"

        return "canvas"

    def _flux_inpaint(
        self,
        image: Image.Image,
        regions: List,
        masks: List[np.ndarray]
    ) -> Image.Image:
        """
        Flux AI inpainting (premium mode).
        Uses diffusion model for high-quality text removal.
        """
        # Initialize Flux inpainter if needed
        if self._flux is None:
            self._flux = FluxInpainter()

        # Check if model can be loaded
        if not self._flux.load():
            logger.warning("Flux model not available, falling back to OpenCV")
            if self._opencv is None:
                self._opencv = OpenCVInpainter("telea")
            return self._opencv.apply(image, regions, masks)

        # Apply Flux inpainting
        try:
            result = self._flux.apply(
                image=image,
                regions=regions,
                masks=masks,
                prompt="clean manga panel, white speech bubble background, no text, seamless",
                negative_prompt="text, letters, words, characters, writing, signature, watermark",
                num_inference_steps=4,  # Fast mode
                strength=0.75,
                guidance_scale=3.5,
            )
            return result

        except Exception as e:
            logger.error(f"Flux inpainting failed: {e}, falling back to OpenCV")
            if self._opencv is None:
                self._opencv = OpenCVInpainter("telea")
            return self._opencv.apply(image, regions, masks)

    def unload_flux(self):
        """Unload Flux model to free memory."""
        if self._flux is not None:
            self._flux.unload()
            self._flux = None


# Singleton instance
_inpainter = None


def get_inpainter(method: str = "auto") -> Inpainter:
    """Get inpainter instance."""
    global _inpainter
    if _inpainter is None or _inpainter.method != method:
        _inpainter = Inpainter(method)
    return _inpainter


def inpaint_image(
    image: Image.Image,
    regions: List,
    masks: List[np.ndarray] = None,
    method: str = "auto",
    mode: str = "realtime"
) -> Image.Image:
    """
    Convenience function for inpainting.

    Args:
        image: Input PIL Image
        regions: List of text regions
        masks: Optional SAM masks
        method: Inpainting method
        mode: Processing mode

    Returns:
        Inpainted image
    """
    inpainter = get_inpainter(method)
    return inpainter.inpaint(image, regions, masks, mode)
