"""
OSB (Outside Speech Bubble) Detection.
Detects text outside bubbles: SFX, narration, onomatopoeia, titles, etc.

Based on MangaTranslator's ocr_detection.py logic:
1. Run OSB YOLO to detect ALL text regions
2. Run Speech Bubble YOLO to detect bubbles
3. Filter: Remove text that overlaps with speech bubbles
4. Return: Only text OUTSIDE bubbles
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """Represents a detected text region."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    region_type: str  # "bubble", "osb", "text_free"
    mask: Optional[np.ndarray] = None

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
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class OSBDetector:
    """
    Detects text outside speech bubbles using YOLO AnimeText model.

    Logic (from MangaTranslator):
    1. Detect ALL text with OSB YOLO (animetext_yolov12x)
    2. Detect speech bubbles with primary YOLO
    3. Filter out text that overlaps with bubbles
    4. Return remaining text = OSB (SFX, titles, etc.)
    """

    def __init__(self):
        self._osb_model = None
        self._bubble_model = None

    def _load_models(self):
        """Lazy load the detection models."""
        if self._osb_model is None:
            from services import Services
            self._osb_model = Services.get_osb_yolo()
            if self._osb_model is None:
                logger.warning("OSB YOLO model not available")

    def _boxes_overlap(self, box1: Tuple, box2: Tuple) -> bool:
        """Check if two bounding boxes overlap (have non-zero intersection)."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        return not (
            x1_max <= x2_min or x2_max <= x1_min or
            y1_max <= y2_min or y2_max <= y1_min
        )

    def _box_is_inside(self, box1: Tuple, box2: Tuple, threshold: float = 0.5) -> bool:
        """Check if box1 is significantly contained inside box2 (IoA > threshold)."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_w = max(0, inter_x_max - inter_x_min)
        inter_h = max(0, inter_y_max - inter_y_min)
        intersection = inter_w * inter_h

        area1 = (x1_max - x1_min) * (y1_max - y1_min)

        if area1 <= 0:
            return False

        ioa = intersection / area1
        return ioa > threshold

    def _filter_nested_detections(self, results: List) -> List:
        """Remove detections fully contained in larger ones to avoid duplicates."""
        if len(results) <= 1:
            return results

        def get_area(result):
            bbox = result[0]
            x_min, y_min, x_max, y_max = bbox
            return (x_max - x_min) * (y_max - y_min)

        sorted_results = sorted(results, key=get_area, reverse=True)
        filtered_results = []

        for current_result in sorted_results:
            is_nested = False
            current_bbox = current_result[0]

            for kept_result in filtered_results:
                kept_bbox = kept_result[0]
                if self._box_is_inside(current_bbox, kept_bbox, threshold=0.9):
                    is_nested = True
                    break

            if not is_nested:
                filtered_results.append(current_result)

        return filtered_results

    def detect(
        self,
        image: Image.Image,
        confidence: float = 0.5,
        bubble_boxes: Optional[List] = None,
        text_free_boxes: Optional[List] = None,
    ) -> List[TextRegion]:
        """
        Detect text regions outside speech bubbles.

        Args:
            image: Input PIL Image
            confidence: Detection confidence threshold
            bubble_boxes: List of existing bubble bboxes [(x1,y1,x2,y2), ...]
            text_free_boxes: Optional list of text_free regions as fallback

        Returns:
            List of TextRegion objects for OSB text
        """
        self._load_models()

        if self._osb_model is None:
            logger.warning("OSB model not loaded, using text_free fallback if available")
            if text_free_boxes:
                return [
                    TextRegion(
                        bbox=tuple(int(c) for c in box[:4]),
                        confidence=0.5,
                        region_type="osb"
                    )
                    for box in text_free_boxes
                ]
            return []

        try:
            # Convert to numpy array (BGR for YOLO)
            img_array = np.array(image)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array

            # Step 1: Run OSB YOLO to detect ALL text regions
            logger.info("Running YOLO OSB Text detection...")
            osb_results = self._osb_model.predict(
                img_bgr,
                conf=confidence,
                verbose=False,
                device='cpu'
            )[0]

            if osb_results.boxes is None or len(osb_results.boxes) == 0:
                logger.info("No text regions detected by OSB YOLO")
                return []

            # Collect all OSB detections
            all_text_detections = []
            for box in osb_results.boxes:
                bbox = tuple(box.xyxy[0].cpu().numpy().astype(int))
                conf = float(box.conf[0])
                all_text_detections.append((bbox, conf))

            logger.info(f"OSB YOLO detected {len(all_text_detections)} text regions")

            # Step 2: Filter out nested detections
            all_text_detections = self._filter_nested_detections(all_text_detections)
            logger.info(f"After filtering nested: {len(all_text_detections)} regions")

            # Step 3: Filter out text that overlaps with speech bubbles
            if bubble_boxes and len(bubble_boxes) > 0:
                filtered_detections = []

                # Normalize bubble boxes format
                normalized_bubbles = []
                for b in bubble_boxes:
                    if isinstance(b, dict):
                        bbox = b.get("bbox")
                        if bbox and len(bbox) == 4:
                            # Could be (x, y, w, h) or (x1, y1, x2, y2)
                            x1, y1, x2_or_w, y2_or_h = bbox
                            # Heuristic: if x2 > x1*2, it's probably (x1,y1,x2,y2)
                            if x2_or_w > x1 and y2_or_h > y1:
                                normalized_bubbles.append((x1, y1, x2_or_w, y2_or_h))
                            else:
                                # (x, y, w, h) format
                                normalized_bubbles.append((x1, y1, x1 + x2_or_w, y1 + y2_or_h))
                    elif hasattr(b, '__iter__') and len(b) >= 4:
                        x1, y1, x2, y2 = b[:4]
                        normalized_bubbles.append((float(x1), float(y1), float(x2), float(y2)))

                for osb_det in all_text_detections:
                    osb_bbox, osb_conf = osb_det
                    overlaps_bubble = False

                    for bubble_bbox in normalized_bubbles:
                        # Check if OSB text overlaps with any bubble
                        if self._boxes_overlap(osb_bbox, bubble_bbox):
                            # Also check if it's inside the bubble
                            if self._box_is_inside(osb_bbox, bubble_bbox, threshold=0.3):
                                overlaps_bubble = True
                                break

                    if not overlaps_bubble:
                        filtered_detections.append(osb_det)

                removed_count = len(all_text_detections) - len(filtered_detections)
                logger.info(f"Filtered out {removed_count} text regions overlapping with bubbles")
                all_text_detections = filtered_detections

            # Step 4: Convert to TextRegion objects
            osb_regions = []
            for bbox, conf in all_text_detections:
                x1, y1, x2, y2 = bbox
                # Filter very small detections
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    osb_regions.append(TextRegion(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf,
                        region_type="osb"
                    ))

            logger.info(f"OSB detection complete: found {len(osb_regions)} text regions outside bubbles")
            return osb_regions

        except Exception as e:
            logger.error(f"OSB detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []


class HybridDetector:
    """
    Combines multiple detectors for comprehensive text detection.
    Orchestrates bubble + OSB + text_free detection.
    """

    def __init__(self):
        self.osb_detector = OSBDetector()

    def detect_all(
        self,
        image: Image.Image,
        bubble_detections: List,
        detect_osb: bool = True,
        osb_confidence: float = 0.5
    ) -> Tuple[List[TextRegion], List[TextRegion]]:
        """
        Detect all text regions in image.

        Args:
            image: Input PIL Image
            bubble_detections: Existing bubble detections (from YOLO)
            detect_osb: Whether to detect OSB text
            osb_confidence: OSB detection confidence threshold

        Returns:
            Tuple of (bubble_regions, osb_regions)
        """
        # Convert bubble detections to TextRegion format
        bubble_regions = []
        bubble_boxes = []  # For OSB filtering

        for det in bubble_detections:
            try:
                if hasattr(det, 'tolist'):
                    det = det.tolist()
                elif hasattr(det, '__iter__') and not isinstance(det, (list, tuple)):
                    det = list(det)

                if isinstance(det, (list, tuple)):
                    if len(det) >= 6:
                        x1, y1, x2, y2, score = det[0], det[1], det[2], det[3], det[4]
                    elif len(det) >= 5:
                        x1, y1, x2, y2, score = det[0], det[1], det[2], det[3], det[4]
                    elif len(det) >= 4:
                        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                        score = 0.5
                    else:
                        continue
                elif hasattr(det, 'bbox'):
                    x1, y1, x2, y2 = det.bbox
                    score = getattr(det, 'confidence', 0.5)
                else:
                    continue

                bubble_regions.append(TextRegion(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(score),
                    region_type="bubble"
                ))
                bubble_boxes.append((float(x1), float(y1), float(x2), float(y2)))

            except Exception as e:
                logger.debug(f"Skipping detection: {e}")
                continue

        # Detect OSB if enabled
        osb_regions = []
        if detect_osb:
            osb_regions = self.osb_detector.detect(
                image,
                confidence=osb_confidence,
                bubble_boxes=bubble_boxes,
            )

        return bubble_regions, osb_regions

    def merge_and_sort(
        self,
        bubble_regions: List[TextRegion],
        osb_regions: List[TextRegion],
        rtl: bool = True
    ) -> List[TextRegion]:
        """
        Merge and sort all regions by reading order.

        Args:
            bubble_regions: List of bubble text regions
            osb_regions: List of OSB text regions
            rtl: Whether to use right-to-left reading order (manga)

        Returns:
            Sorted list of all text regions
        """
        all_regions = bubble_regions + osb_regions

        def sort_key(region: TextRegion):
            cx, cy = region.center
            row = cy // 100
            col = -cx if rtl else cx
            return (row, col)

        return sorted(all_regions, key=sort_key)


# Singleton instance
_hybrid_detector = None


def get_hybrid_detector() -> HybridDetector:
    """Get singleton HybridDetector instance."""
    global _hybrid_detector
    if _hybrid_detector is None:
        _hybrid_detector = HybridDetector()
    return _hybrid_detector
