"""
Bubble detection using YOLOv8 with singleton pattern.
Supports OpenVINO optimization for Intel CPUs.
"""

from typing import List, Tuple, Optional


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _nms(boxes, scores, iou_thresh=0.5):
    if not boxes:
        return []

    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []

    while idxs:
        current = idxs.pop(0)
        keep.append(current)
        idxs = [i for i in idxs if _iou(boxes[current], boxes[i]) < iou_thresh]

    return keep


def detect_bubbles_legacy(model_path, image_path):
    """
    Legacy function - loads model each time (slow).
    Use detect_bubbles() with singleton for better performance.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    bubbles = model(image_path)[0]
    data = bubbles.boxes.data.tolist()

    boxes = [[b[0], b[1], b[2], b[3]] for b in data]
    scores = [b[4] for b in data]
    keep = _nms(boxes, scores, iou_thresh=0.5)

    return [data[i] for i in keep]


def detect_bubbles(image_or_model_path, image_or_model=None, conf: float = 0.25, iou_thresh: float = 0.5) -> List:
    """
    Detect speech bubbles using singleton YOLO model (optimized).

    Backward compatible with old signature: detect_bubbles(model_path, image)
    New signature: detect_bubbles(image) or detect_bubbles(image, model=model)

    Args:
        image_or_model_path: Input image OR model path (for backward compatibility)
        image_or_model: Image (if first arg is model path) OR YOLO model instance
        conf: Confidence threshold for detections
        iou_thresh: IoU threshold for NMS

    Returns:
        List of detections: [(x1, y1, x2, y2, score, class_id), ...]
    """
    # Backward compatibility: detect_bubbles(model_path, image)
    if isinstance(image_or_model_path, str) and image_or_model is not None:
        # Old signature: (model_path, image)
        from services import Services
        model = Services.get_yolo()
        image = image_or_model
    elif isinstance(image_or_model_path, str) and image_or_model is None:
        # Just a file path, use singleton
        from services import Services
        model = Services.get_yolo()
        image = image_or_model_path
    else:
        # New signature: (image) or (image, model=model)
        image = image_or_model_path
        if image_or_model is None:
            from services import Services
            model = Services.get_yolo()
        else:
            model = image_or_model

    results = model(image, conf=conf, verbose=False)[0]
    data = results.boxes.data.tolist()

    if len(data) <= 1:
        return data

    # Apply NMS
    boxes = [[b[0], b[1], b[2], b[3]] for b in data]
    scores = [b[4] for b in data]
    keep = _nms(boxes, scores, iou_thresh=iou_thresh)

    return [data[i] for i in keep]
