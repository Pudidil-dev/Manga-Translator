"""
Unified Translation API.
Single endpoint with dynamic feature toggles.
No modes - users choose exactly what features they want.
"""

import base64
import hashlib
import logging
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from PIL import Image
from flask import Blueprint, jsonify, request

from services import Services

logger = logging.getLogger(__name__)

# Create Blueprint
unified_api = Blueprint('unified_api', __name__)

# ============================================================================
# Caching
# ============================================================================

class TranslationCache:
    """Simple in-memory cache for translations."""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, str] = {}
        self._max_size = max_size

    def get(self, text: str, source: str, target: str) -> Optional[str]:
        key = f"{source}:{target}:{text}"
        return self._cache.get(key)

    def set(self, text: str, translation: str, source: str, target: str):
        if len(self._cache) >= self._max_size:
            # Remove oldest entries
            keys = list(self._cache.keys())[:100]
            for k in keys:
                del self._cache[k]
        key = f"{source}:{target}:{text}"
        self._cache[key] = translation


class ImageCache:
    """Cache for processed images."""

    def __init__(self, max_size: int = 50):
        self._cache: Dict[str, Dict] = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[Dict]:
        return self._cache.get(key)

    def set(self, key: str, data: Dict):
        if len(self._cache) >= self._max_size:
            keys = list(self._cache.keys())[:10]
            for k in keys:
                del self._cache[k]
        self._cache[key] = data


_translation_cache = TranslationCache()
_image_cache = ImageCache()

# ============================================================================
# Helper Functions
# ============================================================================

def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode base64 image to numpy array (RGB)."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]

    image_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compute_image_hash(b64_string: str) -> str:
    """Compute hash for image caching."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    return hashlib.md5(b64_string[:1000].encode()).hexdigest()[:16]


def compute_render_params(bbox: List[int], text: str) -> Dict:
    """Compute optimal text rendering parameters."""
    x, y, w, h = bbox

    # Estimate font size based on bbox
    text_len = len(text) if text else 1
    area = w * h
    font_size = min(max(int(np.sqrt(area / max(text_len, 1)) * 0.8), 10), 24)

    return {
        'font_family': 'Arial, sans-serif',
        'font_size': font_size,
        'align': 'center',
        'line_height': 1.2,
        'padding': 4,
    }


# ============================================================================
# Detection Functions
# ============================================================================

def detect_bubbles(
    image: Image.Image,
    use_sam: bool = False,
    use_advanced: bool = False,
    confidence: float = 0.6
) -> List[Dict]:
    """
    Detect speech bubbles in image.

    Args:
        image: PIL Image
        use_sam: Use SAM2 for precise masks
        use_advanced: Use advanced dual-YOLO detection
        confidence: Detection confidence threshold

    Returns:
        List of detection dicts with bbox, confidence, sam_mask (optional)
    """
    detections = []
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    if use_advanced:
        try:
            from advanced_detection import detect_speech_bubbles
            advanced_dets = detect_speech_bubbles(
                image=image,
                confidence=confidence,
                use_sam=use_sam,
                conjoined_detection=True,
                verbose=False
            )
            for det in advanced_dets:
                bbox = det.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = bbox
                    w, h = x2 - x1, y2 - y1
                    if w > 20 and h > 20:
                        detections.append({
                            'bbox': [x1, y1, w, h],
                            'confidence': det.get('confidence', 0.9),
                            'type': 'bubble',
                            'sam_mask': det.get('sam_mask')
                        })
            return detections
        except Exception as e:
            logger.warning(f"Advanced detection failed, falling back to YOLO: {e}")

    # Standard YOLO detection
    detector = Services.get_yolo()
    if detector is None:
        return []

    results = detector(image_bgr, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w, h = int(x2 - x1), int(y2 - y1)
            if w > 20 and h > 20:
                det = {
                    'bbox': [int(x1), int(y1), w, h],
                    'confidence': float(box.conf[0]),
                    'type': 'bubble'
                }

                # Get SAM mask if requested
                if use_sam:
                    try:
                        sam = Services.get_sam()
                        if sam:
                            mask = sam.segment_box(image_np, [x1, y1, x2, y2])
                            det['sam_mask'] = mask
                    except Exception as e:
                        logger.debug(f"SAM segmentation failed: {e}")

                detections.append(det)

    return detections


def detect_osb(
    image: Image.Image,
    bubble_boxes: List[Tuple],
    confidence: float = 0.4
) -> List[Dict]:
    """
    Detect text outside speech bubbles (SFX, titles, etc).

    Args:
        image: PIL Image
        bubble_boxes: List of bubble bboxes for filtering
        confidence: Detection confidence

    Returns:
        List of OSB detection dicts
    """
    try:
        from osb_detection import get_hybrid_detector
        hybrid = get_hybrid_detector()

        osb_results = hybrid.osb_detector.detect(
            image,
            confidence=confidence,
            bubble_boxes=bubble_boxes,
        )

        detections = []
        for osb in osb_results:
            x1, y1, x2, y2 = osb.bbox
            w, h = x2 - x1, y2 - y1
            if w > 15 and h > 15:
                detections.append({
                    'bbox': [x1, y1, w, h],
                    'confidence': osb.confidence,
                    'type': 'osb'
                })

        return detections
    except Exception as e:
        logger.warning(f"OSB detection failed: {e}")
        return []


# ============================================================================
# OCR & Translation
# ============================================================================

def ocr_region(crop: Image.Image, lang: str = 'ja') -> str:
    """Run OCR on a cropped region."""
    ocr = Services.get_hybrid_ocr()
    if ocr is None:
        return ""

    try:
        text = ocr.read(crop, lang=lang)
        return text.strip() if text else ""
    except Exception as e:
        logger.debug(f"OCR failed: {e}")
        return ""


def translate_batch(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    """
    Batch translate texts efficiently.
    Uses cache for already-translated texts, batches the rest.
    """
    if not texts:
        return []

    results = [''] * len(texts)
    to_translate = []  # (index, text) pairs for texts not in cache

    # Check cache first
    for i, text in enumerate(texts):
        if not text:
            results[i] = ''
            continue
        cached = _translation_cache.get(text, source_lang, target_lang)
        if cached:
            results[i] = cached
        else:
            to_translate.append((i, text))

    # Batch translate uncached texts
    if to_translate:
        translator = Services.get_translator()
        if translator is None:
            # No translator, return original texts
            for i, text in to_translate:
                results[i] = text
        else:
            try:
                texts_to_translate = [t[1] for t in to_translate]
                translations = translator.translate_batch(
                    texts_to_translate,
                    source_lang=source_lang,
                    target_lang=target_lang
                )

                # Map back to results and cache
                for (i, src_text), tgt_text in zip(to_translate, translations):
                    results[i] = tgt_text
                    _translation_cache.set(src_text, tgt_text, source_lang, target_lang)

            except Exception as e:
                logger.warning(f"Batch translation failed: {e}")
                # Fallback to original texts
                for i, text in to_translate:
                    results[i] = text

    return results


# ============================================================================
# Inpainting
# ============================================================================

def apply_inpainting(
    image: np.ndarray,
    detections: List[Dict],
    method: str = 'opencv'
) -> np.ndarray:
    """
    Apply inpainting to remove text from bubbles.

    Args:
        image: Input image (RGB numpy array)
        detections: List of detections with bbox
        method: 'opencv' or 'canvas'

    Returns:
        Inpainted image
    """
    if method == 'canvas':
        return apply_canvas_inpainting(image, detections)
    else:
        return apply_opencv_inpainting(image, detections)


def apply_canvas_inpainting(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Fast canvas overlay - fill bubbles with white, OSB with detected color."""
    result = image.copy()

    for det in detections:
        x, y, w, h = det['bbox']

        # Bounds check
        x, y = max(0, x), max(0, y)
        x2 = min(result.shape[1], x + w)
        y2 = min(result.shape[0], y + h)
        w, h = x2 - x, y2 - y

        if w <= 0 or h <= 0:
            continue

        # For bubbles: use white (standard manga bubble color)
        # For OSB: try to detect background color
        if det.get('type') == 'osb':
            # Sample from corners to detect background
            corner_samples = []
            sample_size = 5
            roi = result[y:y2, x:x2]

            # Top-left corner
            corner_samples.extend(roi[:sample_size, :sample_size].reshape(-1, 3))
            # Top-right corner
            corner_samples.extend(roi[:sample_size, -sample_size:].reshape(-1, 3))
            # Bottom-left corner
            corner_samples.extend(roi[-sample_size:, :sample_size].reshape(-1, 3))
            # Bottom-right corner
            corner_samples.extend(roi[-sample_size:, -sample_size:].reshape(-1, 3))

            if corner_samples:
                fill_color = np.median(corner_samples, axis=0).astype(np.uint8)
            else:
                fill_color = np.array([255, 255, 255], dtype=np.uint8)

            # For OSB: fill rectangle
            result[y:y2, x:x2] = fill_color
        else:
            # For bubbles: white ellipse
            fill_color = np.array([255, 255, 255], dtype=np.uint8)

            # Create ellipse mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w//2, h//2), (int(w*0.45), int(h*0.45)), 0, 0, 360, 255, -1)

            # Apply white fill
            roi = result[y:y2, x:x2]
            roi[mask > 0] = fill_color

    return result


def apply_opencv_inpainting(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """OpenCV TELEA inpainting for cleaner text removal."""
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for det in detections:
        x, y, bw, bh = det['bbox']

        # Use SAM mask if available
        if det.get('sam_mask') is not None:
            sam_mask = det['sam_mask']
            if sam_mask.shape[:2] == (h, w):
                combined_mask = np.maximum(combined_mask, (sam_mask > 0).astype(np.uint8) * 255)
            continue

        # Create ellipse mask
        center_x, center_y = x + bw // 2, y + bh // 2
        axes = (int(bw * 0.4), int(bh * 0.4))
        cv2.ellipse(combined_mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)

    if np.any(combined_mask > 0):
        # Convert to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(image_bgr, combined_mask, 3, cv2.INPAINT_TELEA)
        return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

    return image


# ============================================================================
# Text Rendering
# ============================================================================

def render_text_on_image(
    image: np.ndarray,
    regions: List[Dict]
) -> np.ndarray:
    """Render translated text onto image."""
    from PIL import Image, ImageDraw, ImageFont

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    for region in regions:
        text = region.get('tgt_text', '')
        if not text:
            continue

        bbox = region['bbox']
        x, y, w, h = bbox
        render = region.get('render', {})

        font_size = render.get('font_size', 14)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Simple center text
        center_x = x + w // 2
        center_y = y + h // 2

        # Word wrap
        words = text.split()
        lines = []
        current_line = ""
        max_width = w * 0.85

        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox_test = draw.textbbox((0, 0), test_line, font=font)
            if bbox_test[2] - bbox_test[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        # Draw lines
        line_height = font_size * 1.2
        total_height = len(lines) * line_height
        start_y = center_y - total_height / 2

        for i, line in enumerate(lines):
            bbox_line = draw.textbbox((0, 0), line, font=font)
            text_width = bbox_line[2] - bbox_line[0]
            text_x = center_x - text_width / 2
            text_y = start_y + i * line_height

            draw.text((text_x, text_y), line, fill=(0, 0, 0), font=font)

    return np.array(pil_image)


# ============================================================================
# Main API Endpoint
# ============================================================================

@unified_api.route('/translate', methods=['POST'])
def translate():
    """
    Unified translation endpoint with dynamic feature toggles.

    Request JSON:
    {
        "image_b64": "...",           // Required: Base64 encoded image
        "source_lang": "ja",          // Source language (default: ja)
        "target_lang": "id",          // Target language (default: id)

        // Feature toggles (all optional, defaults shown):
        "use_sam": false,             // Use SAM2 for precise bubble masks
        "use_advanced": false,        // Use advanced dual-YOLO detection
        "detect_osb": true,           // Detect text outside bubbles
        "use_inpainting": true,       // Apply inpainting to remove text
        "inpaint_method": "opencv",   // "opencv" or "canvas"
        "render_text": true,          // Render translated text on image
        "return_image": true,         // Return processed image as base64
    }

    Response JSON:
    {
        "success": true,
        "regions": [...],             // Detected and translated regions
        "image_b64": "...",           // Processed image (if return_image=true)
        "meta": {
            "timings": {...},
            "counts": {...}
        }
    }
    """
    data = request.get_json()
    timings = {}

    # Validate input
    if not data or 'image_b64' not in data:
        return jsonify({'success': False, 'error': 'Missing image_b64'}), 400

    try:
        t0 = time.perf_counter()
        image_np = decode_base64_image(data['image_b64'])
        image_hash = compute_image_hash(data['image_b64'])
        timings['decode_ms'] = int((time.perf_counter() - t0) * 1000)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Invalid image: {e}'}), 400

    # Parse options
    source_lang = data.get('source_lang', 'ja')
    target_lang = data.get('target_lang', 'id')

    use_sam = data.get('use_sam', False)
    use_advanced = data.get('use_advanced', False)
    detect_osb_enabled = data.get('detect_osb', True)
    use_inpainting = data.get('use_inpainting', True)
    inpaint_method = data.get('inpaint_method', 'opencv')
    render_text_enabled = data.get('render_text', True)
    return_image = data.get('return_image', True)

    pil_image = Image.fromarray(image_np)

    # Check cache
    cache_key = f"{image_hash}:{source_lang}:{target_lang}:{use_sam}:{detect_osb_enabled}:{use_inpainting}:{inpaint_method}"
    cached = _image_cache.get(cache_key)
    if cached:
        logger.info(f"Cache hit for {cache_key[:20]}...")
        return jsonify({
            'success': True,
            'regions': cached['regions'],
            'image_b64': cached.get('image_b64'),
            'meta': {**cached['meta'], 'cache_hit': True}
        })

    # 1. Bubble Detection
    t0 = time.perf_counter()
    detections = detect_bubbles(
        pil_image,
        use_sam=use_sam,
        use_advanced=use_advanced
    )
    timings['detection_ms'] = int((time.perf_counter() - t0) * 1000)
    logger.info(f"Detected {len(detections)} bubbles")

    # 2. OSB Detection
    if detect_osb_enabled:
        t0 = time.perf_counter()
        bubble_boxes = [(d['bbox'][0], d['bbox'][1],
                        d['bbox'][0] + d['bbox'][2],
                        d['bbox'][1] + d['bbox'][3]) for d in detections]
        osb_dets = detect_osb(pil_image, bubble_boxes)
        detections.extend(osb_dets)
        timings['osb_ms'] = int((time.perf_counter() - t0) * 1000)
        logger.info(f"Detected {len(osb_dets)} OSB regions")

    # 3. OCR all regions (parallel)
    t0 = time.perf_counter()

    # Prepare crops for OCR
    ocr_data = []  # (detection_index, crop_pil, detection)
    for i, det in enumerate(detections):
        x, y, w, h = det['bbox']
        # Bounds check
        x, y = max(0, x), max(0, y)
        x2 = min(image_np.shape[1], x + w)
        y2 = min(image_np.shape[0], y + h)
        if x2 > x and y2 > y:
            crop = image_np[y:y2, x:x2]
            crop_pil = Image.fromarray(crop)
            ocr_data.append((i, crop_pil, det))

    # Parallel OCR
    ocr = Services.get_hybrid_ocr()
    src_texts = []
    valid_indices = []

    if ocr and ocr_data:
        import concurrent.futures
        import os
        max_workers = max(1, int(os.getenv("OCR_WORKERS", "4")))

        def ocr_one(crop_pil):
            try:
                text = ocr.read(crop_pil, lang=source_lang)
                return text.strip() if text else ""
            except:
                return ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            ocr_results = list(executor.map(ocr_one, [d[1] for d in ocr_data]))

        # Filter valid OCR results
        for (det_idx, crop_pil, det), text in zip(ocr_data, ocr_results):
            if text:
                src_texts.append(text)
                valid_indices.append((det_idx, det))
            elif det.get('type') == 'osb':
                # Keep OSB with placeholder
                src_texts.append("[SFX]")
                valid_indices.append((det_idx, det))

    timings['ocr_ms'] = int((time.perf_counter() - t0) * 1000)
    logger.info(f"OCR completed: {len(src_texts)} texts from {len(ocr_data)} regions")

    # 4. Batch translate all texts at once
    t0 = time.perf_counter()
    tgt_texts = translate_batch(src_texts, source_lang, target_lang)
    timings['translate_ms'] = int((time.perf_counter() - t0) * 1000)
    logger.info(f"Batch translated {len(tgt_texts)} texts")

    # 5. Build regions with translations
    regions = []
    for idx, ((det_idx, det), src_text, tgt_text) in enumerate(zip(valid_indices, src_texts, tgt_texts)):
        render_params = compute_render_params(det['bbox'], tgt_text)
        regions.append({
            'id': f'r{idx}',
            'bbox': det['bbox'],
            'type': det.get('type', 'bubble'),
            'mask_type': 'rect' if det.get('type') == 'osb' else 'ellipse',
            'src_text': src_text,
            'tgt_text': tgt_text,
            'confidence': det.get('confidence', 0.9),
            'render': render_params,
            'has_sam_mask': det.get('sam_mask') is not None
        })

    logger.info(f"Built {len(regions)} text regions")

    # 6. Inpainting (optional)
    result_image = image_np.copy()
    if use_inpainting and regions:
        t0 = time.perf_counter()
        result_image = apply_inpainting(result_image, detections, inpaint_method)
        timings['inpaint_ms'] = int((time.perf_counter() - t0) * 1000)

    # 7. Text Rendering (optional)
    if render_text_enabled and regions:
        t0 = time.perf_counter()
        result_image = render_text_on_image(result_image, regions)
        timings['render_ms'] = int((time.perf_counter() - t0) * 1000)

    # 8. Encode result image
    result_image_b64 = None
    if return_image:
        t0 = time.perf_counter()
        pil_result = Image.fromarray(result_image)
        buffer = BytesIO()
        pil_result.save(buffer, format='PNG')
        result_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        timings['encode_ms'] = int((time.perf_counter() - t0) * 1000)

    # Build response
    meta = {
        'image_hash': image_hash,
        'bubble_count': len([r for r in regions if r['type'] == 'bubble']),
        'osb_count': len([r for r in regions if r['type'] == 'osb']),
        'total_regions': len(regions),
        'features': {
            'sam': use_sam,
            'advanced': use_advanced,
            'osb': detect_osb_enabled,
            'inpainting': use_inpainting,
            'inpaint_method': inpaint_method,
            'render_text': render_text_enabled
        },
        'timings': timings,
        'cache_hit': False
    }

    # Cache result
    _image_cache.set(cache_key, {
        'regions': regions,
        'image_b64': result_image_b64,
        'meta': meta
    })

    total_ms = sum(timings.values())
    logger.info(f"Translation complete: {len(regions)} regions in {total_ms}ms")

    return jsonify({
        'success': True,
        'regions': regions,
        'image_b64': result_image_b64,
        'meta': meta
    })


@unified_api.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    models = Services.get_loaded_models()
    return jsonify({
        'ok': True,
        'models': models
    })


# ============================================================================
# Register Blueprint
# ============================================================================

def register_unified_api(app):
    """Register unified API blueprint with Flask app."""
    app.register_blueprint(unified_api, url_prefix='/api')
    logger.info("Unified API registered at /api/translate")
