"""
Bubble splitting utilities.
Splits large detected regions that may contain multiple speech bubbles.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def split_merged_bubbles(
    image: np.ndarray,
    detections: List[Dict],
    min_bubble_height: int = 60,
    min_bubble_width: int = 40,
    max_aspect_ratio: float = 3.0,
) -> List[Dict]:
    """
    Split detected regions that appear to contain multiple merged bubbles.

    Args:
        image: Original image (RGB numpy array)
        detections: List of detection dicts with 'bbox' and 'confidence'
        min_bubble_height: Minimum height for a valid bubble
        min_bubble_width: Minimum width for a valid bubble
        max_aspect_ratio: Max height/width ratio before considering split

    Returns:
        List of detections, with merged bubbles split into separate regions
    """
    result = []

    for det in detections:
        x, y, w, h = det['bbox']
        confidence = det['confidence']

        # Check if region might contain multiple bubbles
        aspect_ratio = h / w if w > 0 else 0

        # Conditions that suggest multiple bubbles:
        # 1. Very tall region (aspect ratio > 3.0)
        # 2. Height > 2x minimum bubble height
        should_try_split = (
            (aspect_ratio > max_aspect_ratio and h > min_bubble_height * 2) or
            (h > min_bubble_height * 3)
        )

        if should_try_split:
            # Try to split this region
            sub_regions = find_sub_bubbles(image, x, y, w, h, min_bubble_height)

            if len(sub_regions) > 1:
                logger.info(f"Split region [{x},{y},{w},{h}] into {len(sub_regions)} sub-regions")
                for sub_bbox in sub_regions:
                    result.append({
                        'bbox': sub_bbox,
                        'confidence': confidence * 0.95,  # Slightly lower confidence for splits
                        'was_split': True,
                        'type': det.get('type', 'bubble')  # Preserve type field
                    })
            else:
                # Couldn't split, keep original
                result.append(det)
        else:
            result.append(det)

    return result


def find_sub_bubbles(
    image: np.ndarray,
    x: int, y: int, w: int, h: int,
    min_height: int = 60
) -> List[List[int]]:
    """
    Find sub-bubbles within a region by analyzing horizontal white gaps.

    Manga speech bubbles typically have white backgrounds, so we look for
    horizontal strips of white that indicate bubble boundaries.

    Args:
        image: Original image (RGB)
        x, y, w, h: Bounding box of the region
        min_height: Minimum height for a valid sub-bubble

    Returns:
        List of [x, y, w, h] bounding boxes for sub-bubbles
    """
    # Extract region
    crop = image[y:y+h, x:x+w]

    if crop.size == 0:
        return [[x, y, w, h]]

    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # Find white pixels (bubble interior)
    # Threshold for "white" - adjust if needed
    _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Calculate horizontal projection (sum of white pixels per row)
    h_projection = np.sum(white_mask, axis=1) / 255

    # Normalize to percentage of width
    white_ratio = h_projection / w

    # Find rows that are mostly white (potential bubble boundaries)
    # A row with >80% white pixels might be between bubbles or inside a bubble
    # We look for transitions from low-white to high-white regions

    # Find "content" rows (less white = has text/content)
    content_threshold = 0.5  # Less than 50% white = has content
    has_content = white_ratio < content_threshold

    # Find gaps between content regions
    sub_regions = []
    in_content = False
    region_start = 0

    for row_idx in range(len(has_content)):
        if has_content[row_idx] and not in_content:
            # Starting a new content region
            in_content = True
            region_start = row_idx
        elif not has_content[row_idx] and in_content:
            # Ending a content region
            in_content = False
            region_end = row_idx
            region_height = region_end - region_start

            if region_height >= min_height * 0.5:
                # Add some padding
                pad = 10
                sub_y = max(0, region_start - pad)
                sub_h = min(h, region_end + pad) - sub_y

                if sub_h >= min_height * 0.3:
                    sub_regions.append([x, y + sub_y, w, sub_h])

    # Handle case where content extends to bottom
    if in_content:
        region_height = len(has_content) - region_start
        if region_height >= min_height * 0.5:
            pad = 10
            sub_y = max(0, region_start - pad)
            sub_h = h - sub_y
            if sub_h >= min_height * 0.3:
                sub_regions.append([x, y + sub_y, w, sub_h])

    # If we didn't find good splits, try alternative method
    if len(sub_regions) <= 1:
        sub_regions = split_by_contours(image, x, y, w, h, min_height)

    # If still no good splits, return original
    if len(sub_regions) <= 1:
        return [[x, y, w, h]]

    # Merge overlapping regions
    sub_regions = merge_overlapping_regions(sub_regions, y_threshold=20)

    return sub_regions


def split_by_contours(
    image: np.ndarray,
    x: int, y: int, w: int, h: int,
    min_height: int = 60
) -> List[List[int]]:
    """
    Alternative split method using contour detection.
    Finds white bubble regions and returns their bounding boxes.
    """
    # Extract region
    crop = image[y:y+h, x:x+w]

    if crop.size == 0:
        return [[x, y, w, h]]

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [[x, y, w, h]]

    # Filter and convert contours to bounding boxes
    sub_regions = []
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)

        # Filter small contours
        if ch >= min_height * 0.4 and cw >= 30:
            # Convert to global coordinates
            global_bbox = [x + cx, y + cy, cw, ch]
            sub_regions.append(global_bbox)

    # Sort by Y position
    sub_regions.sort(key=lambda r: r[1])

    return sub_regions if sub_regions else [[x, y, w, h]]


def merge_overlapping_regions(
    regions: List[List[int]],
    y_threshold: int = 20
) -> List[List[int]]:
    """
    Merge regions that overlap vertically.
    """
    if len(regions) <= 1:
        return regions

    # Sort by Y position
    regions = sorted(regions, key=lambda r: r[1])

    merged = [regions[0]]

    for current in regions[1:]:
        last = merged[-1]
        last_bottom = last[1] + last[3]
        current_top = current[1]

        # Check if regions overlap or are very close
        if current_top <= last_bottom + y_threshold:
            # Merge: extend the last region
            new_bottom = max(last_bottom, current[1] + current[3])
            new_x = min(last[0], current[0])
            new_right = max(last[0] + last[2], current[0] + current[2])

            merged[-1] = [
                new_x,
                last[1],
                new_right - new_x,
                new_bottom - last[1]
            ]
        else:
            merged.append(current)

    return merged


def filter_duplicate_regions(
    regions: List[Dict],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Remove duplicate/overlapping regions using IoU.
    OSB regions are protected from being filtered by bubble regions.
    """
    if len(regions) <= 1:
        return regions

    def compute_iou(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    # Separate OSB and non-OSB regions
    osb_regions = [r for r in regions if r.get('type') == 'osb']
    other_regions = [r for r in regions if r.get('type') != 'osb']

    # Filter duplicates within non-OSB regions only
    sorted_regions = sorted(other_regions, key=lambda r: r['confidence'], reverse=True)

    keep = []
    for region in sorted_regions:
        is_duplicate = False
        for kept in keep:
            if compute_iou(region['bbox'], kept['bbox']) > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            keep.append(region)

    # Add all OSB regions (don't filter them)
    keep.extend(osb_regions)

    return keep
