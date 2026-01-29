"""
Extension API Routes for Manga-Translator.
Endpoints for browser extension realtime translation.
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
from cache import get_translation_cache

logger = logging.getLogger(__name__)
extension_api = Blueprint('extension_api', __name__)

# Initialize cache
_translation_cache = get_translation_cache(max_memory=2000)


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
            'cache_stats': _translation_cache.stats()
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
            "target_lang": "id"
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

    # 1. Bubble Detection
    t0 = time.perf_counter()
    detector = Services.get_yolo()

    # Convert RGB to BGR for YOLO
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = detector(image_bgr, verbose=False)

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
            'confidence': det['confidence']
        }

    max_workers = min(len(detections), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(ocr_one, det) for det in detections]
        for future in as_completed(futures):
            try:
                result = future.result()
                if result['text']:
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
            'render': render_params
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
