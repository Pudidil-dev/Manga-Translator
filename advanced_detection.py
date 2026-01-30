"""
Advanced Speech Bubble Detection with SAM2 and Dual YOLO.
Ported from MangaTranslator's detection.py for higher accuracy.

Features:
- Dual YOLO detection (primary + secondary for conjoined bubbles)
- SAM2/SAM3 segmentation for precise bubble masks
- IoA/IoU calculations for accurate overlap detection
- Conjoined bubble detection and splitting
- OSB (Outside Speech Bubble) text expansion
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Detection Parameters
IOA_THRESHOLD = 0.50  # 50% IoA threshold for conjoined bubble detection
SAM_MASK_THRESHOLD = 0.5  # SAM2 mask binarization threshold
IOA_OVERLAP_THRESHOLD = 0.5  # IoA threshold for general overlap detection
IOU_DUPLICATE_THRESHOLD = 0.7  # IoU threshold for duplicate primary detection


@dataclass
class Detection:
    """Represents a detected speech bubble or text region."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str
    sam_mask: Optional[np.ndarray] = None
    is_conjoined: bool = False

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with pipeline."""
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class": self.class_name,
            "sam_mask": self.sam_mask,
            "is_conjoined": self.is_conjoined,
        }


def _box_contains(inner: Tuple, outer: Tuple) -> bool:
    """Return True if inner box is fully contained in outer box."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    return ix0 >= ox0 and iy0 >= oy0 and ix1 <= ox1 and iy1 <= oy1


def calculate_ioa(box_inner: Tuple, box_outer: Tuple) -> float:
    """
    Calculate Intersection over Area (IoA) for two bounding boxes.
    IoA = intersection_area / area_of_inner_box

    Args:
        box_inner: (x0, y0, x1, y1) for the inner box
        box_outer: (x0, y0, x1, y1) for the outer box

    Returns:
        IoA value between 0 and 1
    """
    x_inner_min, y_inner_min, x_inner_max, y_inner_max = box_inner
    x_outer_min, y_outer_min, x_outer_max, y_outer_max = box_outer

    inter_x_min = max(x_inner_min, x_outer_min)
    inter_y_min = max(y_inner_min, y_outer_min)
    inter_x_max = min(x_inner_max, x_outer_max)
    inter_y_max = min(y_inner_max, y_outer_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    intersection = inter_w * inter_h

    area_inner = (x_inner_max - x_inner_min) * (y_inner_max - y_inner_min)
    return intersection / area_inner if area_inner > 0 else 0.0


def calculate_iou(box_a: Tuple, box_b: Tuple) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.

    Args:
        box_a: (x0, y0, x1, y1)
        box_b: (x0, y0, x1, y1)

    Returns:
        IoU value between 0 and 1
    """
    inter_x_min = max(box_a[0], box_b[0])
    inter_y_min = max(box_a[1], box_b[1])
    inter_x_max = min(box_a[2], box_b[2])
    inter_y_max = min(box_a[3], box_b[3])

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    intersection = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _deduplicate_boxes(
    boxes: torch.Tensor,
    confidences: torch.Tensor,
    threshold: float
) -> Tuple[torch.Tensor, List[int]]:
    """
    Remove duplicate detections using IoU-based NMS.
    Keeps the box with higher confidence when IoU > threshold.

    Args:
        boxes: Tensor of bounding boxes (N, 4)
        confidences: Tensor of confidence scores (N,)
        threshold: IoU threshold above which boxes are duplicates

    Returns:
        (deduplicated boxes tensor, indices of kept boxes)
    """
    if len(boxes) <= 1:
        return boxes, list(range(len(boxes)))

    boxes_list = boxes.tolist()
    confs_list = confidences.tolist()
    n = len(boxes_list)

    # Sort by confidence (descending)
    indices = sorted(range(n), key=lambda i: confs_list[i], reverse=True)
    keep = []

    for i in indices:
        is_duplicate = False
        for k in keep:
            if calculate_iou(boxes_list[i], boxes_list[k]) > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(i)

    return boxes[keep], keep


def _remove_contained_boxes(boxes: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
    """
    Remove boxes that are fully or almost fully contained within other boxes.

    Args:
        boxes: Tensor of bounding boxes (N, 4)
        threshold: IoA threshold above which inner boxes are removed

    Returns:
        Filtered bounding boxes tensor
    """
    if len(boxes) <= 1:
        return boxes

    boxes_list = boxes.tolist()
    n = len(boxes_list)
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue

            # Check if box i is contained in box j
            if calculate_ioa(boxes_list[i], boxes_list[j]) > threshold:
                keep[i] = False
                break

    return boxes[keep]


def _resolve_mask_overlaps(
    masks: List[np.ndarray],
    boxes: List,
    verbose: bool = False
) -> List[np.ndarray]:
    """
    Resolve overlapping mask regions by carving exclusive boundaries.

    For conjoined bubble masks that overlap:
    1. Finds overlapping pixel regions between mask pairs
    2. Determines split axis based on box arrangement
    3. Assigns overlap pixels to the appropriate mask

    Args:
        masks: List of boolean numpy arrays (SAM output masks)
        boxes: List of corresponding bounding boxes
        verbose: Whether to log operations

    Returns:
        List of resolved masks with non-overlapping regions
    """
    if len(masks) <= 1:
        return masks

    resolved = [mask.copy() for mask in masks]
    n = len(resolved)
    overlap_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            overlap = np.logical_and(resolved[i], resolved[j])
            if not np.any(overlap):
                continue

            overlap_count += 1
            box_a = boxes[i] if not hasattr(boxes[i], "tolist") else boxes[i].tolist()
            box_b = boxes[j] if not hasattr(boxes[j], "tolist") else boxes[j].tolist()

            center_a = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
            center_b = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
            overlap_coords = np.where(overlap)

            # Perpendicular bisector split via cross product
            mid_x = (center_a[0] + center_b[0]) / 2
            mid_y = (center_a[1] + center_b[1]) / 2
            vec_x = center_b[0] - center_a[0]
            vec_y = center_b[1] - center_a[1]

            pixel_y = overlap_coords[0]
            pixel_x = overlap_coords[1]
            cross = (pixel_x - mid_x) * vec_y - (pixel_y - mid_y) * vec_x

            mask_a_pixels = cross >= 0
            mask_b_pixels = cross < 0

            resolved[i][pixel_y[mask_b_pixels], pixel_x[mask_b_pixels]] = False
            resolved[j][pixel_y[mask_a_pixels], pixel_x[mask_a_pixels]] = False

    if overlap_count > 0 and verbose:
        logger.info(f"Resolved {overlap_count} overlapping mask region(s)")

    return resolved


def _categorize_detections(
    primary_boxes: torch.Tensor,
    secondary_boxes: torch.Tensor,
    ioa_threshold: float = IOA_THRESHOLD
) -> Tuple[List[Tuple[int, List[int]]], List[int]]:
    """
    Categorize detections into simple and conjoined bubbles.

    Args:
        primary_boxes: Primary YOLO detection boxes (N, 4)
        secondary_boxes: Secondary YOLO detection boxes (M, 4)
        ioa_threshold: Threshold for determining containment

    Returns:
        (conjoined_indices, simple_indices)
        - conjoined_indices: List of (primary_idx, [secondary_indices])
        - simple_indices: List of primary indices that are simple bubbles
    """
    # Handle single detection edge cases
    if primary_boxes.ndim == 1 and primary_boxes.numel() == 4:
        primary_boxes = primary_boxes.unsqueeze(0)
    if secondary_boxes.ndim == 1 and secondary_boxes.numel() == 4:
        secondary_boxes = secondary_boxes.unsqueeze(0)

    conjoined_indices = []
    processed_secondary_indices = set()

    for i, p_box in enumerate(primary_boxes):
        contained_indices = []
        for j, s_box in enumerate(secondary_boxes):
            if j in processed_secondary_indices:
                continue
            ioa = calculate_ioa(s_box.tolist(), p_box.tolist())
            if ioa > ioa_threshold:
                contained_indices.append(j)

        if len(contained_indices) >= 2:
            conjoined_indices.append((i, contained_indices))
            processed_secondary_indices.update(contained_indices)

    primary_simple_indices = []
    conjoined_primary_indices = {c[0] for c in conjoined_indices}

    for i in range(len(primary_boxes)):
        if i in conjoined_primary_indices:
            continue

        # Check for duplication against processed secondary bubbles
        is_duplicate = False
        p_box_list = primary_boxes[i].tolist()

        for s_idx in processed_secondary_indices:
            s_box_list = secondary_boxes[s_idx].tolist()
            if calculate_ioa(s_box_list, p_box_list) > ioa_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            primary_simple_indices.append(i)

    return conjoined_indices, primary_simple_indices


def _expand_boxes_with_osb_text(
    image: np.ndarray,
    primary_boxes: torch.Tensor,
    osb_model,
    confidence: float,
    verbose: bool = False
) -> torch.Tensor:
    """
    Expand speech-bubble boxes to fully contain detected OSB text boxes.

    Args:
        image: Input image as numpy array (BGR)
        primary_boxes: Primary detection boxes tensor
        osb_model: OSB YOLO model
        confidence: Detection confidence threshold
        verbose: Whether to log details

    Returns:
        Expanded primary boxes tensor
    """
    if primary_boxes is None or len(primary_boxes) == 0:
        return primary_boxes

    if osb_model is None:
        return primary_boxes

    try:
        osb_results = osb_model(image, conf=confidence, verbose=False, device='cpu')[0]
        osb_boxes = (
            osb_results.boxes.xyxy
            if osb_results.boxes is not None
            else torch.tensor([])
        )

        if osb_boxes is None or len(osb_boxes) == 0:
            return primary_boxes

        pb_np = primary_boxes.detach().cpu().numpy()
        osb_np = osb_boxes.detach().cpu().numpy()

        for t_box in osb_np:
            tx0, ty0, tx1, ty1 = t_box
            best_idx = None
            best_intersection = 0.0

            for i, b_box in enumerate(pb_np):
                bx0, by0, bx1, by1 = b_box
                inter_x0 = max(bx0, tx0)
                inter_y0 = max(by0, ty0)
                inter_x1 = min(bx1, tx1)
                inter_y1 = min(by1, ty1)
                inter_w = max(0.0, inter_x1 - inter_x0)
                inter_h = max(0.0, inter_y1 - inter_y0)
                intersection = inter_w * inter_h
                if intersection > best_intersection:
                    best_intersection = intersection
                    best_idx = i

            if best_idx is None or best_intersection <= 0.0:
                continue

            if _box_contains(t_box, pb_np[best_idx]):
                continue

            bx0, by0, bx1, by1 = pb_np[best_idx]
            pb_np[best_idx] = [
                min(bx0, tx0),
                min(by0, ty0),
                max(bx1, tx1),
                max(by1, ty1),
            ]

        return torch.tensor(
            pb_np, device=primary_boxes.device, dtype=primary_boxes.dtype
        )

    except Exception as e:
        if verbose:
            logger.warning(f"OSB text verification skipped: {e}")
        return primary_boxes


def _process_simple_bubbles_with_sam(
    image: Image.Image,
    boxes: torch.Tensor,
    indices: List[int],
    sam_predictor,
    device: torch.device
) -> List[np.ndarray]:
    """
    Process simple (non-conjoined) speech bubbles using SAM2.

    Args:
        image: PIL Image
        boxes: Tensor of bounding boxes
        indices: List of indices for simple bubbles
        sam_predictor: SAM2 predictor instance
        device: PyTorch device

    Returns:
        List of numpy boolean masks for simple bubbles
    """
    if not indices:
        return []

    try:
        # Set image for SAM predictor
        image_np = np.array(image.convert("RGB"))
        sam_predictor.set_image(image_np)

        masks = []
        boxes_np = boxes[indices].cpu().numpy()

        for box in boxes_np:
            # SAM2 expects box as [x1, y1, x2, y2]
            mask, _, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False
            )
            # mask is (1, H, W), take first mask
            masks.append(mask[0] > SAM_MASK_THRESHOLD)

        return masks

    except Exception as e:
        logger.warning(f"SAM processing failed: {e}")
        return []


class AdvancedDetector:
    """
    Advanced speech bubble detector using dual YOLO + SAM2.
    Provides more accurate detection than simple YOLO alone.
    """

    def __init__(self):
        self._primary_model = None
        self._secondary_model = None
        self._osb_model = None
        self._sam_predictor = None
        self._device = None

    def _load_models(self, use_sam: bool = True):
        """Lazy load detection models."""
        from services import Services

        if self._primary_model is None:
            self._primary_model = Services.get_yolo()

        if self._secondary_model is None:
            self._secondary_model = Services.get_conjoined_yolo()

        if self._osb_model is None:
            self._osb_model = Services.get_osb_yolo()

        if use_sam and self._sam_predictor is None:
            self._sam_predictor = Services.get_sam()

        # Detect device
        if self._device is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

    def detect(
        self,
        image: Image.Image,
        confidence: float = 0.6,
        conjoined_confidence: float = 0.35,
        use_sam: bool = True,
        conjoined_detection: bool = True,
        osb_text_verification: bool = False,
        verbose: bool = False
    ) -> Tuple[List[Detection], List[Tuple]]:
        """
        Detect speech bubbles using dual YOLO models and SAM2.

        Args:
            image: Input PIL Image
            confidence: Confidence threshold for primary YOLO
            conjoined_confidence: Confidence threshold for secondary YOLO
            use_sam: Whether to use SAM for mask refinement
            conjoined_detection: Whether to detect conjoined bubbles
            osb_text_verification: Whether to expand boxes with OSB text
            verbose: Whether to log detailed information

        Returns:
            (detections, text_free_boxes)
            - detections: List of Detection objects
            - text_free_boxes: List of text-free box tuples
        """
        self._load_models(use_sam=use_sam)

        detections = []
        text_free_boxes = []

        if self._primary_model is None:
            logger.error("Primary YOLO model not available")
            return detections, text_free_boxes

        # Convert image to numpy
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        img_h, img_w = image_cv.shape[:2]

        # Run primary YOLO detection
        try:
            primary_results = self._primary_model(
                image_cv, conf=confidence, verbose=False, device='cpu'
            )[0]
            primary_boxes = (
                primary_results.boxes.xyxy
                if primary_results.boxes is not None
                else torch.tensor([])
            )
        except Exception as e:
            logger.error(f"Primary YOLO detection failed: {e}")
            return detections, text_free_boxes

        if len(primary_boxes) == 0:
            if verbose:
                logger.info("No primary detections found")
            return detections, text_free_boxes

        # Deduplicate primary detections
        if len(primary_boxes) > 1:
            original_count = len(primary_boxes)
            primary_boxes, _ = _deduplicate_boxes(
                primary_boxes, primary_results.boxes.conf, IOU_DUPLICATE_THRESHOLD
            )
            if verbose and len(primary_boxes) < original_count:
                logger.info(f"Removed {original_count - len(primary_boxes)} duplicate detections")

        # Remove contained boxes
        if len(primary_boxes) > 1:
            original_count = len(primary_boxes)
            primary_boxes = _remove_contained_boxes(primary_boxes)
            if verbose and len(primary_boxes) < original_count:
                logger.info(f"Removed {original_count - len(primary_boxes)} contained detections")

        logger.info(f"Detected {len(primary_boxes)} speech bubbles with primary YOLO")

        # Run secondary YOLO for conjoined detection
        secondary_boxes = torch.tensor([])
        if conjoined_detection and self._secondary_model is not None:
            try:
                secondary_results = self._secondary_model(
                    image_cv, conf=conjoined_confidence, verbose=False, device='cpu'
                )[0]
                secondary_boxes = (
                    secondary_results.boxes.xyxy
                    if secondary_results.boxes is not None
                    else torch.tensor([])
                )

                if len(secondary_boxes) > 1:
                    secondary_boxes = _remove_contained_boxes(secondary_boxes)

                if verbose and len(secondary_boxes) > 0:
                    logger.info(f"Secondary YOLO found {len(secondary_boxes)} detections")

            except Exception as e:
                logger.warning(f"Secondary YOLO detection failed: {e}")

        # OSB text verification (expand boxes)
        if osb_text_verification and self._osb_model is not None:
            primary_boxes = _expand_boxes_with_osb_text(
                image_cv, primary_boxes, self._osb_model, confidence, verbose
            )

        # Categorize into simple vs conjoined bubbles
        conjoined_indices = []
        simple_indices = list(range(len(primary_boxes)))

        if use_sam and len(secondary_boxes) > 0 and conjoined_detection:
            conjoined_indices, simple_indices = _categorize_detections(
                primary_boxes, secondary_boxes, IOA_THRESHOLD
            )
            if conjoined_indices:
                logger.info(f"Found {len(conjoined_indices)} conjoined bubble groups")

        # Process with SAM if available
        if use_sam and self._sam_predictor is not None:
            try:
                # Collect all boxes to process
                boxes_to_process = []
                conjoined_mask_ranges = []

                for idx in simple_indices:
                    boxes_to_process.append(primary_boxes[idx])

                for _, s_indices in conjoined_indices:
                    start_idx = len(boxes_to_process)
                    for s_idx in s_indices:
                        boxes_to_process.append(secondary_boxes[s_idx])
                    end_idx = len(boxes_to_process)
                    conjoined_mask_ranges.append((start_idx, end_idx))

                if boxes_to_process:
                    all_boxes_tensor = torch.stack(boxes_to_process)
                    all_masks = _process_simple_bubbles_with_sam(
                        image,
                        all_boxes_tensor,
                        list(range(len(boxes_to_process))),
                        self._sam_predictor,
                        self._device
                    )

                    # Resolve overlaps in conjoined groups
                    for start_idx, end_idx in conjoined_mask_ranges:
                        if start_idx < len(all_masks) and end_idx <= len(all_masks):
                            group_masks = all_masks[start_idx:end_idx]
                            group_boxes = boxes_to_process[start_idx:end_idx]
                            resolved_masks = _resolve_mask_overlaps(
                                group_masks, group_boxes, verbose=verbose
                            )
                            all_masks[start_idx:end_idx] = resolved_masks

                    # Create Detection objects with SAM masks
                    for i, (mask, box) in enumerate(zip(all_masks, boxes_to_process)):
                        x0_f, y0_f, x1_f, y1_f = box.tolist()

                        x0 = int(np.floor(max(0, min(x0_f, img_w))))
                        y0 = int(np.floor(max(0, min(y0_f, img_h))))
                        x1 = int(np.ceil(max(0, min(x1_f, img_w))))
                        y1 = int(np.ceil(max(0, min(y1_f, img_h))))

                        if x1 <= x0 or y1 <= y0:
                            continue

                        # Clip mask to bbox
                        bbox_mask = np.zeros((img_h, img_w), dtype=bool)
                        bbox_mask[y0:y1, x0:x1] = True
                        clipped_mask = np.logical_and(mask, bbox_mask)

                        is_conjoined = any(
                            start_idx <= i < end_idx
                            for start_idx, end_idx in conjoined_mask_ranges
                        )

                        detection = Detection(
                            bbox=(x0, y0, x1, y1),
                            confidence=1.0,  # SAM masks are high confidence
                            class_name="speech_bubble",
                            sam_mask=clipped_mask.astype(np.uint8) * 255,
                            is_conjoined=is_conjoined
                        )
                        detections.append(detection)

                    logger.info(f"SAM segmentation completed: {len(detections)} masks")
                    return detections, text_free_boxes

            except Exception as e:
                logger.warning(f"SAM processing failed: {e}. Falling back to YOLO boxes.")

        # Fallback: Use YOLO boxes without SAM masks
        for i, box in enumerate(primary_boxes):
            x0_f, y0_f, x1_f, y1_f = box.tolist()

            try:
                conf = float(primary_results.boxes.conf[i])
            except (IndexError, AttributeError):
                conf = confidence

            detection = Detection(
                bbox=(
                    int(round(x0_f)),
                    int(round(y0_f)),
                    int(round(x1_f)),
                    int(round(y1_f)),
                ),
                confidence=conf,
                class_name="speech_bubble",
                sam_mask=None,
                is_conjoined=False
            )
            detections.append(detection)

        logger.info(f"YOLO detection completed: {len(detections)} bubbles")
        return detections, text_free_boxes


# Singleton instance
_advanced_detector = None


def get_advanced_detector() -> AdvancedDetector:
    """Get singleton AdvancedDetector instance."""
    global _advanced_detector
    if _advanced_detector is None:
        _advanced_detector = AdvancedDetector()
    return _advanced_detector


def detect_speech_bubbles(
    image: Image.Image,
    confidence: float = 0.6,
    conjoined_confidence: float = 0.35,
    use_sam: bool = True,
    conjoined_detection: bool = True,
    osb_text_verification: bool = False,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Convenience function for detecting speech bubbles.
    Returns detections in dict format compatible with existing pipeline.

    Args:
        image: Input PIL Image
        confidence: Confidence threshold for primary YOLO
        conjoined_confidence: Confidence threshold for secondary YOLO
        use_sam: Whether to use SAM for mask refinement
        conjoined_detection: Whether to detect conjoined bubbles
        osb_text_verification: Whether to expand boxes with OSB text
        verbose: Whether to log detailed information

    Returns:
        List of detection dictionaries with keys:
        - bbox: (x1, y1, x2, y2)
        - confidence: float
        - class: str
        - sam_mask: np.ndarray or None
    """
    detector = get_advanced_detector()
    detections, _ = detector.detect(
        image=image,
        confidence=confidence,
        conjoined_confidence=conjoined_confidence,
        use_sam=use_sam,
        conjoined_detection=conjoined_detection,
        osb_text_verification=osb_text_verification,
        verbose=verbose
    )

    return [det.to_dict() for det in detections]
