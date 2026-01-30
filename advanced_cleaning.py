"""
Advanced Bubble Cleaning with Flux Inpainting.
Ported from MangaTranslator's cleaning.py and inpainting.py for higher quality.

Features:
- Advanced mask processing with SAM masks
- Colored bubble detection and classification
- Flux Kontext/Klein AI inpainting for text removal
- OpenCV TELEA/NS inpainting fallback
- Otsu threshold retry for difficult cases
"""

import logging
from typing import List, Tuple, Optional, Dict, Any, Union, Literal
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Cleaning parameters
GRAYSCALE_MIDPOINT = 128  # Threshold for black vs white bubbles
MIN_CONTOUR_AREA = 50  # Minimum area for filtering small contours
DILATION_KERNEL_SIZE = (7, 7)
EROSION_KERNEL_SIZE = (5, 5)
DISTANCE_TRANSFORM_MASK_SIZE = 5

# Classification thresholds for colored bubbles
BRIGHT_RATIO_THRESHOLD = 0.50
DARK_RATIO_THRESHOLD = 0.50
BRIGHT_DOM_RATIO_MIN = 0.30
DARK_DOM_RATIO_MIN = 0.30
BRIGHT_DARK_RATIO_MAX = 0.10
DARK_BRIGHT_RATIO_MAX = 0.10


@dataclass
class ProcessedBubble:
    """Result of processing a single speech bubble."""
    mask: np.ndarray  # Validated text mask (0/255)
    base_mask: np.ndarray  # Normalized detection mask
    color: Tuple[int, int, int]  # BGR fill color
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    is_colored: bool  # Whether bubble is colored (not white/black)
    text_bbox: Optional[Tuple[int, int, int, int]] = None
    is_sam: bool = False
    inpainted: bool = False


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is uint8 binary (0/255)."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def process_single_bubble(
    base_mask: np.ndarray,
    img_gray: np.ndarray,
    img_height: int,
    img_width: int,
    thresholding_value: int = 190,
    use_otsu_threshold: bool = False,
    roi_shrink_px: int = 4,
    verbose: bool = False,
    detection_bbox: Optional[Tuple] = None,
    is_sam: bool = False,
    dilation_kernel: Optional[np.ndarray] = None,
    erosion_kernel: Optional[np.ndarray] = None,
    min_contour_area: float = MIN_CONTOUR_AREA,
    classify_colored: bool = False
) -> Tuple[np.ndarray, Tuple[int, int, int], bool, Tuple[int, int, int], Optional[Tuple]]:
    """
    Process a single speech bubble mask to extract text regions and determine fill color.

    Args:
        base_mask: Base mask (SAM or YOLO) for the bubble
        img_gray: Grayscale image
        img_height: Image height
        img_width: Image width
        thresholding_value: Fixed threshold for text detection
        use_otsu_threshold: Whether to use Otsu's method
        roi_shrink_px: Pixels to shrink ROI inwards
        verbose: Whether to log details
        detection_bbox: Bounding box for logging
        is_sam: Whether this is a SAM mask
        dilation_kernel: Kernel for morphological dilation
        erosion_kernel: Kernel for morphological erosion
        min_contour_area: Minimum area for contours
        classify_colored: Whether to detect colored bubbles

    Returns:
        (final_mask, fill_color_bgr, is_colored, sample_color_bgr, text_bbox)

    Raises:
        ValueError: If processing fails
    """
    base_mask = _normalize_mask(base_mask)

    if dilation_kernel is None:
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, DILATION_KERNEL_SIZE
        )
    if erosion_kernel is None:
        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, EROSION_KERNEL_SIZE
        )

    masked_pixels = img_gray[base_mask == 255]
    if masked_pixels.size == 0:
        raise ValueError(f"Empty mask for detection {detection_bbox}")

    mean_pixel_value = np.mean(masked_pixels)
    is_black_bubble = mean_pixel_value < GRAYSCALE_MIDPOINT
    fill_color_bgr = (0, 0, 0) if is_black_bubble else (255, 255, 255)
    is_colored_bubble = False
    sample_color_bgr = fill_color_bgr

    if verbose:
        bubble_type = "Black" if is_black_bubble else "White"
        mask_type = "[SAM]" if is_sam else ""
        logger.info(f"{mask_type}Detection {detection_bbox}: {bubble_type} bubble (mean={mean_pixel_value:.1f})")

    # Dilate mask to create ROI
    roi_mask = cv2.dilate(base_mask, dilation_kernel, iterations=1)
    roi_gray = np.zeros_like(img_gray)
    roi_indices = roi_mask == 255
    roi_gray[roi_indices] = img_gray[roi_indices]

    # Invert for black bubbles
    roi_for_thresholding = (
        cv2.bitwise_not(roi_gray) if is_black_bubble else roi_gray
    )
    thresholded_roi = np.zeros_like(img_gray)

    if use_otsu_threshold:
        roi_pixels_for_otsu = roi_for_thresholding[roi_indices]
        thresh_val, _ = cv2.threshold(
            roi_pixels_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if verbose:
            logger.info(f"  Otsu threshold: {thresh_val}")
        _, thresholded_roi = cv2.threshold(
            roi_for_thresholding, thresh_val, 255, cv2.THRESH_BINARY
        )
    else:
        _, thresholded_roi = cv2.threshold(
            roi_for_thresholding, thresholding_value, 255, cv2.THRESH_BINARY
        )

    thresholded_roi = cv2.bitwise_and(thresholded_roi, roi_mask)

    # Shrink ROI to avoid border artifacts
    dist_map = cv2.distanceTransform(roi_mask, cv2.DIST_L2, DISTANCE_TRANSFORM_MASK_SIZE)
    shrunk_roi_mask = np.where(dist_map >= float(roi_shrink_px), 255, 0).astype(np.uint8)
    thresholded_roi = cv2.bitwise_and(thresholded_roi, shrunk_roi_mask)

    # Eroded constraint mask to avoid erasing bubble outlines
    eroded_constraint_mask = cv2.erode(base_mask, erosion_kernel, iterations=1)

    # Find and filter contours
    contours, _ = cv2.findContours(
        thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= min_contour_area:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        if (
            0 <= cx < img_width
            and 0 <= cy < img_height
            and eroded_constraint_mask[cy, cx] == 255
        ):
            valid_contours.append(cnt)

    if verbose:
        logger.info(f"Detection {detection_bbox}: {len(valid_contours)} text fragments found")

    text_bbox = None
    if valid_contours:
        validated_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.drawContours(validated_mask, valid_contours, -1, 255, thickness=cv2.FILLED)

        # Re-contour to get clean boundary
        boundary_contours, _ = cv2.findContours(
            validated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if boundary_contours:
            largest_contour = max(boundary_contours, key=cv2.contourArea)
            final_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            x, y, w, h = cv2.boundingRect(largest_contour)
            text_bbox = (x, y, x + w, y + h)

            if classify_colored:
                # Sample bubble interior to determine if colored
                sampling_mask = cv2.erode(base_mask, erosion_kernel, iterations=2)
                if text_bbox:
                    x1, y1, x2, y2 = text_bbox
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)
                    sampling_mask[y1:y2, x1:x2] = 0

                sample_pixels = img_gray[sampling_mask == 255]
                if sample_pixels.size == 0:
                    sample_pixels = masked_pixels

                sample_values = sample_pixels.astype(np.uint8).flatten()
                hist = np.bincount(sample_values, minlength=256)
                dominant_val = int(hist.argmax()) if hist.size > 0 else int(mean_pixel_value)
                dominant_count = int(hist.max()) if hist.size > 0 else 0
                total_count = max(int(sample_values.size), 1)
                dominant_ratio = dominant_count / float(total_count)
                bright_ratio = float(np.count_nonzero(sample_values >= 245)) / float(total_count)
                dark_ratio = float(np.count_nonzero(sample_values <= 15)) / float(total_count)

                if bright_ratio >= BRIGHT_RATIO_THRESHOLD or (
                    dominant_val >= 245
                    and dominant_ratio >= BRIGHT_DOM_RATIO_MIN
                    and dark_ratio <= BRIGHT_DARK_RATIO_MAX
                ):
                    is_colored_bubble = False
                    fill_color_bgr = (255, 255, 255)
                    sample_color_bgr = (255, 255, 255)
                elif dark_ratio >= DARK_RATIO_THRESHOLD or (
                    dominant_val <= 15
                    and dominant_ratio >= DARK_DOM_RATIO_MIN
                    and bright_ratio <= DARK_BRIGHT_RATIO_MAX
                ):
                    is_colored_bubble = False
                    fill_color_bgr = (0, 0, 0)
                    sample_color_bgr = (0, 0, 0)
                else:
                    is_colored_bubble = True
                    sample_color_bgr = (dominant_val, dominant_val, dominant_val)
                    if verbose:
                        logger.info(f"Detection {detection_bbox}: colored/gradient bubble")

            return (final_mask, fill_color_bgr, is_colored_bubble, sample_color_bgr, text_bbox)

    raise ValueError(f"Failed to process bubble mask for {detection_bbox}")


class AdvancedCleaner:
    """
    Advanced bubble cleaning with Flux AI inpainting support.
    """

    def __init__(self, inpaint_method: Literal["auto", "canvas", "opencv", "flux"] = "auto"):
        self.inpaint_method = inpaint_method
        self._flux_inpainter = None
        self._opencv_inpainter = None

    def clean(
        self,
        image: Image.Image,
        detections: List[Dict],
        thresholding_value: int = 190,
        use_otsu_threshold: bool = False,
        roi_shrink_px: int = 4,
        inpaint_colored: bool = False,
        verbose: bool = False,
        mode: str = "realtime"
    ) -> Tuple[Image.Image, List[ProcessedBubble]]:
        """
        Clean speech bubbles from image.

        Args:
            image: Input PIL Image
            detections: List of detection dicts with bbox and optional sam_mask
            thresholding_value: Threshold for text detection
            use_otsu_threshold: Whether to use Otsu's method
            roi_shrink_px: Pixels to shrink ROI
            inpaint_colored: Whether to use AI inpainting for colored bubbles
            verbose: Whether to log details
            mode: Processing mode (realtime, quality, premium)

        Returns:
            (cleaned_image, processed_bubbles)
        """
        # Convert to numpy
        image_np = np.array(image.convert("RGB"))
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        img_height, img_width = image_cv.shape[:2]

        cleaned_image = image_cv.copy()
        processed_bubbles = []

        # Prepare kernels
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DILATION_KERNEL_SIZE)
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, EROSION_KERNEL_SIZE)

        for detection in detections:
            try:
                sam_mask = detection.get("sam_mask")
                bbox = detection.get("bbox")

                if sam_mask is not None:
                    base_mask = _normalize_mask(sam_mask)
                    is_sam = True
                else:
                    # Create rectangular mask from bbox
                    x1, y1, x2, y2 = bbox
                    base_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    base_mask[y1:y2, x1:x2] = 255
                    is_sam = False

                # Process bubble
                try:
                    final_mask, fill_color, is_colored, sample_color, text_bbox = process_single_bubble(
                        base_mask=base_mask,
                        img_gray=img_gray,
                        img_height=img_height,
                        img_width=img_width,
                        thresholding_value=thresholding_value,
                        use_otsu_threshold=use_otsu_threshold,
                        roi_shrink_px=roi_shrink_px,
                        verbose=verbose,
                        detection_bbox=bbox,
                        is_sam=is_sam,
                        dilation_kernel=dilation_kernel,
                        erosion_kernel=erosion_kernel,
                        classify_colored=inpaint_colored
                    )

                    bubble = ProcessedBubble(
                        mask=final_mask,
                        base_mask=base_mask,
                        color=sample_color if sample_color else fill_color,
                        bbox=bbox,
                        is_colored=is_colored,
                        text_bbox=text_bbox,
                        is_sam=is_sam,
                        inpainted=False
                    )
                    processed_bubbles.append(bubble)

                except ValueError as e:
                    # Try with Otsu if not already using it
                    if not use_otsu_threshold:
                        if verbose:
                            logger.info(f"Retrying with Otsu for {bbox}")
                        try:
                            final_mask, fill_color, is_colored, sample_color, text_bbox = process_single_bubble(
                                base_mask=base_mask,
                                img_gray=img_gray,
                                img_height=img_height,
                                img_width=img_width,
                                thresholding_value=thresholding_value,
                                use_otsu_threshold=True,  # Force Otsu
                                roi_shrink_px=roi_shrink_px,
                                verbose=verbose,
                                detection_bbox=bbox,
                                is_sam=is_sam,
                                dilation_kernel=dilation_kernel,
                                erosion_kernel=erosion_kernel,
                                classify_colored=inpaint_colored
                            )

                            bubble = ProcessedBubble(
                                mask=final_mask,
                                base_mask=base_mask,
                                color=sample_color if sample_color else fill_color,
                                bbox=bbox,
                                is_colored=is_colored,
                                text_bbox=text_bbox,
                                is_sam=is_sam,
                                inpainted=False
                            )
                            processed_bubbles.append(bubble)
                        except ValueError:
                            if verbose:
                                logger.warning(f"Failed to process bubble {bbox}")
                    else:
                        if verbose:
                            logger.warning(f"Failed to process bubble {bbox}: {e}")

            except Exception as e:
                logger.error(f"Error processing detection {detection.get('bbox')}: {e}")
                continue

        # Apply inpainting based on method and mode
        if processed_bubbles:
            cleaned_image = self._apply_inpainting(
                cleaned_image,
                processed_bubbles,
                mode=mode,
                inpaint_colored=inpaint_colored,
                verbose=verbose
            )

        # Convert back to PIL
        cleaned_pil = Image.fromarray(cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB))

        logger.info(f"Cleaned {len(processed_bubbles)} speech bubbles")
        return cleaned_pil, processed_bubbles

    def _apply_inpainting(
        self,
        image: np.ndarray,
        bubbles: List[ProcessedBubble],
        mode: str,
        inpaint_colored: bool,
        verbose: bool
    ) -> np.ndarray:
        """Apply inpainting to cleaned image."""
        method = self._select_method(mode)

        if method == "canvas":
            return self._apply_canvas_inpaint(image, bubbles)
        elif method == "opencv":
            return self._apply_opencv_inpaint(image, bubbles)
        elif method == "flux":
            return self._apply_flux_inpaint(image, bubbles, inpaint_colored, verbose)
        else:
            return self._apply_canvas_inpaint(image, bubbles)

    def _select_method(self, mode: str) -> str:
        """Select inpainting method based on mode."""
        if self.inpaint_method != "auto":
            return self.inpaint_method

        if mode == "realtime":
            return "canvas"
        elif mode == "quality":
            return "opencv"
        elif mode == "premium":
            return "flux"
        return "canvas"

    def _apply_canvas_inpaint(
        self,
        image: np.ndarray,
        bubbles: List[ProcessedBubble]
    ) -> np.ndarray:
        """Fast canvas overlay inpainting."""
        result = image.copy()

        # Group by color for efficiency
        color_groups = {}
        for bubble in bubbles:
            if bubble.inpainted:
                continue
            color_key = bubble.color
            if color_key not in color_groups:
                color_groups[color_key] = []
            color_groups[color_key].append(bubble.mask)

        for color_bgr, masks in color_groups.items():
            combined_mask = np.bitwise_or.reduce(masks)
            if result.shape[2] == 4:
                result[combined_mask == 255, :3] = color_bgr
            else:
                result[combined_mask == 255] = color_bgr

        return result

    def _apply_opencv_inpaint(
        self,
        image: np.ndarray,
        bubbles: List[ProcessedBubble],
        radius: int = 3
    ) -> np.ndarray:
        """OpenCV TELEA inpainting."""
        result = image.copy()

        # Combine all masks
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for bubble in bubbles:
            if bubble.inpainted:
                continue
            combined_mask = np.maximum(combined_mask, bubble.mask)

        if np.any(combined_mask > 0):
            result = cv2.inpaint(result, combined_mask, radius, cv2.INPAINT_TELEA)

        return result

    def _apply_flux_inpaint(
        self,
        image: np.ndarray,
        bubbles: List[ProcessedBubble],
        inpaint_colored: bool,
        verbose: bool
    ) -> np.ndarray:
        """Flux AI inpainting for premium/quality mode when flux method is selected."""
        result = image.copy()

        # Get all bubbles that need inpainting
        bubbles_to_inpaint = [b for b in bubbles if not b.inpainted]

        if not bubbles_to_inpaint:
            return result

        try:
            from inpainting import FluxInpainter

            # Initialize Flux inpainter if needed
            if self._flux_inpainter is None:
                self._flux_inpainter = FluxInpainter()

            # Load model (will skip if already loaded)
            if not self._flux_inpainter.load():
                logger.warning("Flux model not available, falling back to OpenCV")
                return self._apply_opencv_inpaint(image, bubbles_to_inpaint)

            # Convert to PIL for Flux
            pil_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

            # Prepare regions for Flux (convert ProcessedBubble to bbox tuples)
            regions = []
            for bubble in bubbles_to_inpaint:
                x1, y1, x2, y2 = bubble.bbox
                regions.append((x1, y1, x2, y2))

            # Collect SAM masks if available
            masks = [b.mask if b.is_sam else None for b in bubbles_to_inpaint]

            # Apply Flux inpainting
            logger.info(f"Applying Flux inpainting to {len(regions)} regions...")
            inpainted_pil = self._flux_inpainter.apply(
                image=pil_image,
                regions=regions,
                masks=masks,
                prompt="clean manga panel, white speech bubble background, no text, seamless",
                negative_prompt="text, letters, words, characters, writing, signature, watermark",
                num_inference_steps=4,
                strength=0.75,
                guidance_scale=3.5,
            )

            # Mark bubbles as inpainted
            for bubble in bubbles_to_inpaint:
                bubble.inpainted = True

            result = cv2.cvtColor(np.array(inpainted_pil), cv2.COLOR_RGB2BGR)
            logger.info("Flux inpainting completed successfully")

        except ImportError as e:
            logger.warning(f"Flux dependencies not installed: {e}, using OpenCV fallback")
            result = self._apply_opencv_inpaint(image, bubbles_to_inpaint)

        except Exception as e:
            logger.error(f"Flux inpainting failed: {e}, using OpenCV fallback")
            import traceback
            traceback.print_exc()
            result = self._apply_opencv_inpaint(image, bubbles_to_inpaint)

        return result


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
            # Get bbox (handle both ProcessedBubble and raw tuple)
            if hasattr(region, 'bbox'):
                x1, y1, x2, y2 = region.bbox
            elif len(region) >= 4:
                x1, y1, x2, y2 = int(region[0]), int(region[1]), int(region[2]), int(region[3])
            else:
                continue

            if fill_color:
                color = fill_color
            else:
                color = self._detect_background(image, (x1, y1, x2, y2))

            draw.rectangle([x1, y1, x2, y2], fill=color)

        return result

    def _detect_background(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int]:
        """Detect dominant background color from border pixels."""
        x1, y1, x2, y2 = bbox
        img_array = np.array(image)

        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        h, w = img_array.shape[:2]
        border_pixels = []
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
            return (255, 255, 255)

        border_array = np.array(border_pixels)
        median_color = np.median(border_array, axis=0).astype(int)

        return tuple(median_color)


class OpenCVInpainter:
    """OpenCV-based inpainting for quality mode."""

    def __init__(self, algorithm: Literal["telea", "ns"] = "telea"):
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
        """Apply OpenCV inpainting to remove text."""
        img_array = np.array(image)

        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        h, w = img_array.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for i, region in enumerate(regions):
            if masks and i < len(masks) and masks[i] is not None:
                mask = masks[i]
                if mask.shape[:2] == (h, w):
                    combined_mask = np.maximum(combined_mask, mask.astype(np.uint8) * 255)
                else:
                    resized_mask = cv2.resize(mask.astype(np.uint8), (w, h))
                    combined_mask = np.maximum(combined_mask, resized_mask * 255)
            else:
                if hasattr(region, 'bbox'):
                    x1, y1, x2, y2 = region.bbox
                elif hasattr(region, 'mask'):
                    combined_mask = np.maximum(combined_mask, region.mask)
                    continue
                elif len(region) >= 4:
                    x1, y1, x2, y2 = int(region[0]), int(region[1]), int(region[2]), int(region[3])
                else:
                    continue

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                combined_mask[y1:y2, x1:x2] = 255

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


# Singleton instance
_advanced_cleaner = None


def get_advanced_cleaner(method: str = "auto") -> AdvancedCleaner:
    """Get singleton AdvancedCleaner instance."""
    global _advanced_cleaner
    if _advanced_cleaner is None or _advanced_cleaner.inpaint_method != method:
        _advanced_cleaner = AdvancedCleaner(method)
    return _advanced_cleaner


def clean_speech_bubbles(
    image: Image.Image,
    detections: List[Dict],
    thresholding_value: int = 190,
    use_otsu_threshold: bool = False,
    roi_shrink_px: int = 4,
    inpaint_method: str = "auto",
    inpaint_colored: bool = False,
    verbose: bool = False,
    mode: str = "realtime"
) -> Tuple[Image.Image, List[Dict]]:
    """
    Convenience function for cleaning speech bubbles.

    Args:
        image: Input PIL Image
        detections: List of detection dicts
        thresholding_value: Threshold for text detection
        use_otsu_threshold: Whether to use Otsu's method
        roi_shrink_px: Pixels to shrink ROI
        inpaint_method: Inpainting method (auto, canvas, opencv, flux)
        inpaint_colored: Whether to use AI inpainting for colored bubbles
        verbose: Whether to log details
        mode: Processing mode

    Returns:
        (cleaned_image, processed_bubbles as dicts)
    """
    cleaner = get_advanced_cleaner(inpaint_method)
    cleaned_image, bubbles = cleaner.clean(
        image=image,
        detections=detections,
        thresholding_value=thresholding_value,
        use_otsu_threshold=use_otsu_threshold,
        roi_shrink_px=roi_shrink_px,
        inpaint_colored=inpaint_colored,
        verbose=verbose,
        mode=mode
    )

    # Convert ProcessedBubble to dict
    bubble_dicts = []
    for b in bubbles:
        bubble_dicts.append({
            "mask": b.mask,
            "base_mask": b.base_mask,
            "color": b.color,
            "bbox": b.bbox,
            "is_colored": b.is_colored,
            "text_bbox": b.text_bbox,
            "is_sam": b.is_sam,
            "inpainted": b.inpainted
        })

    return cleaned_image, bubble_dicts
