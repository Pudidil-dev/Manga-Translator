"""
Extension API Routes for Manga-Translator.
Endpoints for browser extension realtime translation.

V1 API: Fast translation with basic detection
V2 API: Advanced translation with SAM2, dual YOLO, OSB detection
"""

import base64
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from PIL import Image

from services import Services
from cache import get_translation_cache, get_v2_image_cache
from bubble_split import split_merged_bubbles, filter_duplicate_regions

# Try to import advanced detection
try:
    from advanced_detection import detect_speech_bubbles, get_advanced_detector
    ADVANCED_DETECTION_AVAILABLE = True
except ImportError:
    ADVANCED_DETECTION_AVAILABLE = False

logger = logging.getLogger(__name__)
extension_api = Blueprint('extension_api', __name__)

# Initialize caches
_translation_cache = get_translation_cache(max_memory=2000)
_v2_image_cache = get_v2_image_cache(max_items=100, max_size_mb=500)


def decode_base64_image(image_b64: str) -> np.ndarray:
    """Decode base64 image to numpy array (RGB)."""
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]

    image_data = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return np.array(image)


def compute_image_hash(image_b64: str) -> str:
    """Compute short hash of image for caching."""
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]
    return f"img_{hashlib.sha256(image_b64.encode()).hexdigest()[:16]}"


def compute_render_params(bbox: List[int], text: str, mask_type: str = "ellipse") -> dict:
    """Compute font size and render parameters for bubble."""
    x, y, w, h = bbox
    text_len = max(len(text), 1)

    # Calculate optimal font size based on area
    area_mult = 0.70 if mask_type == "ellipse" else 0.90
    available_area = w * h * area_mult
    font_size = int(min(
        max(10, (available_area / text_len / 0.7) ** 0.5),
        w / 4,
        h / 2.5,
        28
    ))

    # Scale up for short text
    if text_len <= 3:
        font_size = min(int(font_size * 1.4), int(h * 0.5))
    elif text_len <= 8:
        font_size = min(int(font_size * 1.2), int(h * 0.4))

    return {
        "mask_mode": "solid_white",
        "font_family": "sans-serif",
        "font_size": max(10, font_size),
        "align": "center",
        "line_height": 1.25,
        "padding": max(4, int(min(w, h) * 0.06))
    }


@extension_api.route('/health')
def health():
    """Health check endpoint for extension."""
    try:
        yolo = Services.get_yolo()
        ocr = Services.get_hybrid_ocr()
        translator = Services.get_translator()

        return jsonify({
            'ok': True,
            'model_loaded': yolo is not None,
            'detector_loaded': yolo is not None,
            'ocr_loaded': ocr is not None,
            'translator_loaded': translator is not None,
            'yolo_format': Services.get_yolo_format(),
            'cache_stats': _translation_cache.stats(),
            'v2_cache_stats': _v2_image_cache.stats(),
            # Combined Architecture info
            'advanced_detection': ADVANCED_DETECTION_AVAILABLE,
            'models': Services.get_loaded_models(),
            'gpu_info': Services.get_gpu_info(),
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500


@extension_api.route('/v1/translate_viewport', methods=['POST'])
def translate_viewport():
    """
    Single image translation endpoint.

    Request:
        {
            "image_b64": "data:image/png;base64,...",
            "source_lang": "ja",
            "target_lang": "id",
            "use_advanced": false,  // Optional: use advanced SAM2 detection
            "detect_osb": true      // Optional: detect text outside bubbles (SFX, titles)
        }

    Response:
        {
            "regions": [...],
            "meta": {...}
        }
    """
    data = request.json
    timings = {}

    # Validate input
    if not data or 'image_b64' not in data:
        return jsonify({'error': 'Missing image_b64'}), 400

    try:
        image_np = decode_base64_image(data['image_b64'])
        image_hash = compute_image_hash(data['image_b64'])
    except Exception as e:
        return jsonify({'error': f'Invalid image: {e}'}), 400

    source_lang = data.get('source_lang', 'ja')
    target_lang = data.get('target_lang', 'id')
    use_advanced = data.get('use_advanced', False)
    detect_osb = data.get('detect_osb', False)

    # 1. Bubble Detection
    t0 = time.perf_counter()

    # Use advanced detection if requested and available
    if use_advanced and ADVANCED_DETECTION_AVAILABLE:
        pil_image = Image.fromarray(image_np)
        advanced_dets = detect_speech_bubbles(
            image=pil_image,
            confidence=0.6,
            use_sam=False,  # Disable SAM for speed in V1
            conjoined_detection=False,
            verbose=False
        )
        detections = []
        for det in advanced_dets:
            bbox = det.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                if w > 20 and h > 20:
                    detections.append({
                        'bbox': [x, y, w, h],
                        'confidence': det.get('confidence', 0.9)
                    })
        timings['detection_type'] = 'advanced'
    else:
        # Use standard YOLO detection
        detector = Services.get_yolo()
        if detector is None:
            return jsonify({'error': 'YOLO model not loaded'}), 500

        # Convert RGB to BGR for YOLO
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = detector(image_bgr, verbose=False, device='cpu')

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                if w > 20 and h > 20:  # Filter tiny detections
                    detections.append({
                        'bbox': [x, y, w, h],
                        'confidence': float(box.conf[0])
                    })
        timings['detection_type'] = 'yolo'

    # Add OSB detection if enabled
    if detect_osb:
        try:
            from osb_detection import get_hybrid_detector
            pil_image = Image.fromarray(image_np)
            hybrid = get_hybrid_detector()

            # Extract bubble boxes for OSB filtering
            bubble_boxes = []
            for det in detections:
                if det.get('type') != 'osb':  # Only use bubble detections
                    x, y, w, h = det['bbox']
                    bubble_boxes.append((x, y, x + w, y + h))  # Convert to (x1,y1,x2,y2)

            osb_results = hybrid.osb_detector.detect(
                pil_image,
                confidence=0.4,  # Lower threshold for title/SFX text
                bubble_boxes=bubble_boxes,  # Pass bubble boxes for filtering
            )
            for osb in osb_results:
                x1, y1, x2, y2 = osb.bbox
                w, h = x2 - x1, y2 - y1
                if w > 15 and h > 15:  # Filter very small detections
                    detections.append({
                        'bbox': [x1, y1, w, h],
                        'confidence': osb.confidence,
                        'type': 'osb'
                    })
            timings['osb_count'] = len(osb_results)
            logger.info(f"OSB detection: found {len(osb_results)} text regions outside bubbles")
        except Exception as e:
            logger.warning(f"OSB detection failed: {e}")
            import traceback
            traceback.print_exc()
            timings['osb_error'] = str(e)

    timings['detect_ms'] = int((time.perf_counter() - t0) * 1000)

    if not detections:
        return jsonify({
            'regions': [],
            'meta': {
                'image_hash': image_hash,
                'regions_count': 0,
                **timings,
                'ocr_ms': 0,
                'translate_ms': 0,
                'cached_hits': 0
            }
        })

    # 1.5. Split merged bubbles (tall regions that may contain multiple bubbles)
    t0_split = time.perf_counter()
    detections = split_merged_bubbles(image_np, detections)
    detections = filter_duplicate_regions(detections, iou_threshold=0.5)
    timings['split_ms'] = int((time.perf_counter() - t0_split) * 1000)
    logger.info(f"After split: {len(detections)} regions")

    # 2. OCR per region (parallel)
    t0 = time.perf_counter()
    ocr = Services.get_hybrid_ocr()
    ocr_results = []

    def ocr_one(det):
        x, y, w, h = det['bbox']
        crop = image_np[y:y+h, x:x+w]
        crop_pil = Image.fromarray(crop)
        text = ocr.read(crop_pil, lang=source_lang)
        return {
            'bbox': det['bbox'],
            'text': text.strip() if text else '',
            'confidence': det['confidence'],
            'type': det.get('type', 'bubble')  # Preserve type field
        }

    max_workers = min(len(detections), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(ocr_one, det) for det in detections]
        for future in as_completed(futures):
            try:
                result = future.result()
                # Keep OSB regions even if OCR fails (stylized text)
                # Keep bubble regions only if OCR succeeds
                if result['text'] or result.get('type') == 'osb':
                    ocr_results.append(result)
            except Exception as e:
                logger.warning(f"OCR failed: {e}")

    timings['ocr_ms'] = int((time.perf_counter() - t0) * 1000)

    if not ocr_results:
        return jsonify({
            'regions': [],
            'meta': {
                'image_hash': image_hash,
                'regions_count': 0,
                **timings,
                'translate_ms': 0,
                'cached_hits': 0
            }
        })

    # 3. Batch Translation with cache
    t0 = time.perf_counter()
    src_texts = [r['text'] for r in ocr_results]

    cached, uncached_texts, uncached_indices = _translation_cache.get_batch(
        src_texts, source_lang, target_lang
    )

    translations = [None] * len(src_texts)
    for i, trans in cached.items():
        translations[i] = trans

    if uncached_texts:
        translator = Services.get_translator()
        new_translations = translator.translate_batch(
            uncached_texts,
            source_lang=source_lang,
            target_lang=target_lang
        )

        for i, idx in enumerate(uncached_indices):
            trans = new_translations[i] if i < len(new_translations) else uncached_texts[i]
            translations[idx] = trans
            _translation_cache.set(uncached_texts[i], trans, source_lang, target_lang)

    timings['translate_ms'] = int((time.perf_counter() - t0) * 1000)
    cached_hits = len(cached)

    # 4. Build response
    regions = []
    for i, ocr_result in enumerate(ocr_results):
        tgt_text = translations[i] or ocr_result['text']
        render_params = compute_render_params(ocr_result['bbox'], tgt_text)

        regions.append({
            'id': f'r{i}',
            'bbox': ocr_result['bbox'],
            'polygon': [],
            'mask_type': 'ellipse',
            'src_text': ocr_result['text'],
            'tgt_text': tgt_text,
            'confidence': ocr_result['confidence'],
            'render': render_params,
            'type': ocr_result.get('type', 'bubble')  # Include type in response
        })

    return jsonify({
        'regions': regions,
        'meta': {
            'image_hash': image_hash,
            'regions_count': len(regions),
            **timings,
            'cached_hits': cached_hits
        }
    })


@extension_api.route('/v1/batch_detect', methods=['POST'])
def batch_detect():
    """
    Batch detect bubbles + OCR for multiple images.

    Request:
        {
            "images": [
                {"image_id": "hash1", "image_b64": "..."},
                {"image_id": "hash2", "image_b64": "..."}
            ],
            "source_lang": "ja"
        }
    """
    data = request.json
    images = data.get('images', [])
    source_lang = data.get('source_lang', 'ja')

    if not images:
        return jsonify({
            'bubbles': [],
            'total_images': 0,
            'total_bubbles': 0,
            'detect_ms': 0,
            'ocr_ms': 0
        })

    detector = Services.get_yolo()
    ocr = Services.get_hybrid_ocr()

    all_bubbles = []
    total_detect = 0
    total_ocr = 0

    def process_single(item):
        try:
            image_np = decode_base64_image(item['image_b64'])
            image_id = item['image_id']
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Detect
            t0 = time.perf_counter()
            results = detector(image_bgr, verbose=False)
            detect_time = time.perf_counter() - t0

            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    if w > 20 and h > 20:
                        detections.append({
                            'bbox': [x, y, w, h],
                            'confidence': float(box.conf[0])
                        })

            # OCR
            t0 = time.perf_counter()
            bubbles = []
            for i, det in enumerate(detections):
                x, y, w, h = det['bbox']
                crop = image_np[y:y+h, x:x+w]
                crop_pil = Image.fromarray(crop)
                text = ocr.read(crop_pil, lang=source_lang)

                if text and text.strip():
                    bubbles.append({
                        'image_id': image_id,
                        'bubble_id': f"{image_id}_b{i}",
                        'bbox': det['bbox'],
                        'polygon': [],
                        'mask_type': 'ellipse',
                        'src_text': text.strip(),
                        'confidence': det['confidence']
                    })

            ocr_time = time.perf_counter() - t0
            return bubbles, detect_time, ocr_time

        except Exception as e:
            logger.error(f"Error processing {item.get('image_id', 'unknown')}: {e}")
            return [], 0, 0

    # Process in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single, img) for img in images]
        for future in as_completed(futures):
            bubbles, detect_time, ocr_time = future.result()
            all_bubbles.extend(bubbles)
            total_detect += detect_time
            total_ocr += ocr_time

    return jsonify({
        'bubbles': all_bubbles,
        'total_images': len(images),
        'total_bubbles': len(all_bubbles),
        'detect_ms': int(total_detect * 1000),
        'ocr_ms': int(total_ocr * 1000)
    })


@extension_api.route('/v1/batch_translate', methods=['POST'])
def batch_translate():
    """
    Batch translate bubbles with optional glossary.

    Request:
        {
            "bubbles": [...],
            "glossary": {"田中": "Tanaka"},
            "source_lang": "ja",
            "target_lang": "id"
        }
    """
    data = request.json
    bubbles = data.get('bubbles', [])
    source_lang = data.get('source_lang', 'ja')
    target_lang = data.get('target_lang', 'id')

    if not bubbles:
        return jsonify({
            'bubbles': [],
            'total_translated': 0,
            'translate_ms': 0,
            'cached_hits': 0
        })

    # Sort by reading order (top to bottom, right to left for manga)
    sorted_bubbles = sorted(bubbles, key=lambda b: (b['bbox'][1], -b['bbox'][0]))

    # Extract texts
    src_texts = [b['src_text'] for b in sorted_bubbles]

    # Check cache
    t0 = time.perf_counter()
    cached, uncached_texts, uncached_indices = _translation_cache.get_batch(
        src_texts, source_lang, target_lang
    )

    # Translate uncached
    translations = [None] * len(src_texts)
    for i, trans in cached.items():
        translations[i] = trans

    if uncached_texts:
        translator = Services.get_translator()
        new_translations = translator.translate_batch(
            uncached_texts,
            source_lang=source_lang,
            target_lang=target_lang
        )

        for i, idx in enumerate(uncached_indices):
            trans = new_translations[i] if i < len(new_translations) else uncached_texts[i]
            translations[idx] = trans
            _translation_cache.set(uncached_texts[i], trans, source_lang, target_lang)

    translate_ms = int((time.perf_counter() - t0) * 1000)

    # Build response (maintain original order)
    bubble_translations = {
        sorted_bubbles[i]['bubble_id']: translations[i]
        for i in range(len(sorted_bubbles))
    }

    translated_bubbles = []
    for bubble in bubbles:
        tgt_text = bubble_translations.get(bubble['bubble_id'], bubble['src_text'])
        render_params = compute_render_params(bubble['bbox'], tgt_text)

        translated_bubbles.append({
            'image_id': bubble['image_id'],
            'bubble_id': bubble['bubble_id'],
            'bbox': bubble['bbox'],
            'polygon': bubble.get('polygon', []),
            'mask_type': bubble.get('mask_type', 'ellipse'),
            'src_text': bubble['src_text'],
            'tgt_text': tgt_text,
            'confidence': bubble.get('confidence', 0.9),
            'render': render_params
        })

    return jsonify({
        'bubbles': translated_bubbles,
        'total_translated': len(translated_bubbles),
        'translate_ms': translate_ms,
        'cached_hits': len(cached)
    })


@extension_api.route('/v1/cache/stats', methods=['GET'])
def cache_stats():
    """Get cache statistics."""
    return jsonify(_translation_cache.stats())


@extension_api.route('/v1/cache/clear', methods=['POST'])
def cache_clear():
    """Clear translation cache."""
    _translation_cache.clear()
    return jsonify({'ok': True, 'message': 'Cache cleared'})


# =============================================================================
# V2 API - Combined Architecture Endpoints
# =============================================================================

@extension_api.route('/v2/translate', methods=['POST'])
def translate_v2():
    """
    V2 Translation endpoint with Combined Architecture support.
    Includes caching for expensive inpainting operations.

    Request:
        {
            "image_b64": "data:image/png;base64,...",
            "source_lang": "ja",
            "target_lang": "id",
            "mode": "realtime" | "quality" | "premium",
            "detect_osb": true,
            "inpaint_method": "auto" | "canvas" | "opencv" | "flux"
        }

    Response:
        {
            "regions": [...],
            "image_b64": "..." (optional, for quality/premium modes),
            "meta": {...}
        }
    """
    data = request.json
    timings = {}

    # Validate input
    if not data or 'image_b64' not in data:
        return jsonify({'error': 'Missing image_b64'}), 400

    try:
        image_b64 = data['image_b64']
        image_np = decode_base64_image(image_b64)
        image_hash = compute_image_hash(image_b64)
    except Exception as e:
        return jsonify({'error': f'Invalid image: {e}'}), 400

    # Get options
    source_lang = data.get('source_lang', 'ja')
    target_lang = data.get('target_lang', 'id')
    mode = data.get('mode', 'realtime')
    detect_osb = data.get('detect_osb', True)
    inpaint_method = data.get('inpaint_method', 'auto')

    # Check V2 cache first
    cache_key = _v2_image_cache.make_key(
        image_hash, mode, source_lang, target_lang, detect_osb, inpaint_method
    )
    cached_result = _v2_image_cache.get(cache_key)
    if cached_result:
        logger.info(f"V2 cache hit for {cache_key[:12]}...")
        return jsonify(cached_result)

    # Convert to PIL
    pil_image = Image.fromarray(image_np)

    # Use the new pipeline
    from pipeline import MangaPipeline

    pipeline = MangaPipeline(
        source_lang=source_lang,
        target_lang=target_lang,
        mode=mode,
        detect_osb=detect_osb,
        inpaint_method=inpaint_method,
    )

    try:
        t0 = time.perf_counter()
        result_image, timing = pipeline.translate_image(pil_image)
        timings['total_ms'] = int((time.perf_counter() - t0) * 1000)

        # Build regions from pipeline results
        regions = _extract_regions_from_pipeline(
            pil_image, pipeline, source_lang, target_lang
        )

        # Encode result image if not realtime
        result_image_b64 = None
        if mode != "realtime":
            buffer = BytesIO()
            result_image.save(buffer, format='PNG')
            result_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        meta = {
            'image_hash': image_hash,
            'mode': mode,
            'regions_count': len(regions),
            'cache_hit': False,
            **timing,
            **timings,
        }

        # Store in V2 cache
        _v2_image_cache.set(cache_key, result_image_b64, regions, meta)
        logger.info(f"V2 cached result for {cache_key[:12]}... ({timings.get('total_ms', 0)}ms)")

        return jsonify({
            'regions': regions,
            'image_b64': result_image_b64,
            'meta': meta
        })

    except Exception as e:
        logger.error(f"V2 translation failed: {e}")
        return jsonify({'error': str(e)}), 500


def _extract_regions_from_pipeline(image, pipeline, source_lang, target_lang):
    """Extract region info from pipeline for V2 response."""
    image_np = np.array(image)
    ocr = Services.get_hybrid_ocr()
    detections = []

    # Use advanced detection if available
    if ADVANCED_DETECTION_AVAILABLE and pipeline.use_advanced:
        advanced_dets = detect_speech_bubbles(
            image=image,
            confidence=pipeline.config.detection.bubble_confidence,
            use_sam=pipeline.use_sam,
            conjoined_detection=pipeline.conjoined_detection,
            verbose=False
        )
        for det in advanced_dets:
            bbox = det.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                if w > 20 and h > 20:
                    detections.append({
                        'bbox': [x, y, w, h],
                        'confidence': det.get('confidence', 0.9),
                        'type': 'bubble',
                        'sam_mask': det.get('sam_mask')
                    })
    else:
        # Fallback to standard YOLO
        detector = Services.get_yolo()
        if detector is None:
            return []

        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = detector(image_bgr, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                if w > 20 and h > 20:
                    detections.append({
                        'bbox': [x, y, w, h],
                        'confidence': float(box.conf[0]),
                        'type': 'bubble'
                    })

    # OSB Detection
    if pipeline.detect_osb:
        try:
            from osb_detection import get_hybrid_detector
            hybrid = get_hybrid_detector()

            # Extract bubble boxes for OSB filtering
            bubble_boxes = []
            for det in detections:
                if det.get('type') == 'bubble':
                    x, y, w, h = det['bbox']
                    bubble_boxes.append((x, y, x + w, y + h))  # Convert to (x1,y1,x2,y2)

            osb_results = hybrid.osb_detector.detect(
                image,
                confidence=0.5,
                bubble_boxes=bubble_boxes,  # Pass bubble boxes for filtering
            )
            for osb in osb_results:
                x1, y1, x2, y2 = osb.bbox
                detections.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'confidence': osb.confidence,
                    'type': 'osb'
                })
        except Exception as e:
            logger.warning(f"OSB detection in V2 failed: {e}")
            import traceback
            traceback.print_exc()

    # OCR + Translate
    regions = []
    for i, det in enumerate(detections):
        x, y, w, h = det['bbox']
        crop = image_np[y:y+h, x:x+w]
        crop_pil = Image.fromarray(crop)

        src_text = ocr.read(crop_pil, lang=source_lang)

        # Keep OSB regions even if OCR fails (stylized text)
        # Skip bubble regions only if OCR returns empty
        if not src_text or not src_text.strip():
            if det.get('type') != 'osb':
                continue
            # For OSB with no OCR text, use placeholder
            src_text = "[SFX]"

        # Check cache
        cached, _, _ = _translation_cache.get_batch(
            [src_text.strip()], source_lang, target_lang
        )

        if 0 in cached:
            tgt_text = cached[0]
        else:
            translator = Services.get_translator()
            translations = translator.translate_batch(
                [src_text.strip()],
                source_lang=source_lang,
                target_lang=target_lang
            )
            tgt_text = translations[0] if translations else src_text.strip()
            _translation_cache.set(src_text.strip(), tgt_text, source_lang, target_lang)

        render_params = compute_render_params(det['bbox'], tgt_text)

        regions.append({
            'id': f'r{i}',
            'bbox': det['bbox'],
            'polygon': [],
            'mask_type': 'ellipse' if det['type'] == 'bubble' else 'rect',
            'type': det['type'],
            'src_text': src_text.strip(),
            'tgt_text': tgt_text,
            'confidence': det['confidence'],
            'render': render_params
        })

    return regions


@extension_api.route('/v2/models/preload', methods=['POST'])
def preload_models():
    """
    Preload models for a specific mode.

    Request:
        {
            "mode": "realtime" | "quality" | "premium",
            "source_lang": "ja"
        }
    """
    data = request.json or {}
    mode = data.get('mode', 'realtime')
    source_lang = data.get('source_lang', 'ja')

    try:
        Services.preload_for_mode(mode, source_lang)
        return jsonify({
            'ok': True,
            'mode': mode,
            'models': Services.get_loaded_models()
        })
    except Exception as e:
        logger.error(f"Model preload failed: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


@extension_api.route('/v2/models/status', methods=['GET'])
def models_status():
    """Get status of all loaded models."""
    return jsonify({
        'models': Services.get_loaded_models(),
        'gpu_info': Services.get_gpu_info()
    })
