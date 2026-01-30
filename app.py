from flask import Flask, render_template, request, redirect
from flask_cors import CORS
from detect_bubbles import detect_bubbles
from process_bubble import process_bubble
from add_text import add_text
from services import Services
from api_extension import extension_api
from api_unified import register_unified_api
from PIL import Image
import numpy as np
import base64
import cv2
import os
import concurrent.futures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "secret_key")

# Enable CORS for all APIs
CORS(app, resources={
    r"/health": {"origins": "*"},
    r"/v1/*": {"origins": "*"},
    r"/v2/*": {"origins": "*"},
    r"/api/*": {"origins": "*"}  # Unified API
})

# Register API blueprints
app.register_blueprint(extension_api)  # Legacy V1/V2 APIs
register_unified_api(app)  # New unified API at /api/translate

MODEL_PATH = "model/model.pt"

# Preload models at startup for faster first request
# Using try/except to handle potential version compatibility issues
logger.info("Preloading models...")
try:
    Services.preload_all()
    logger.info("Models loaded!")
except Exception as e:
    logger.warning(f"Preload failed (will load on first request): {e}")
    logger.warning("If you see numpy errors, try: pip install 'numpy<2.0'")
LANGUAGE_MAP = {
    "english": "en",
    "indonesian": "id",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "portuguese": "pt",
    "russian": "ru",
    "arabic": "ar",
    "thai": "th",
    "vietnamese": "vi",
    "chinese (simplified)": "zh",
    "chinese (traditional)": "zh-tw",
    "korean": "ko",
    "japanese": "ja",
    "malay": "ms",
    "hindi": "hi",
    "italian": "it",
    "dutch": "nl",
    "polish": "pl",
    "turkish": "tr",
}

# Source language map for OCR engine selection
SOURCE_LANGUAGE_MAP = {
    "japanese": "ja",
    "korean": "ko",
    "chinese (simplified)": "zh",
    "chinese (traditional)": "zh-tw",
    "english": "en",
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def upload_file():
    # Get source language (for OCR engine selection)
    selected_source = request.form.get("source_language", "Japanese").lower()
    source_lang = SOURCE_LANGUAGE_MAP.get(selected_source, "ja")

    # Get target language (for translation)
    selected_language = request.form.get("selected_language", "English").lower()
    target_lang = LANGUAGE_MAP.get(selected_language, "en")

    selected_font = request.form.get("selected_font", "animeace").lower()
    if selected_font == "animeace":
        selected_font += "_"

    if "file" in request.files:
        file = request.files["file"]
        name = file.filename.split(".")[0]

        file_stream = file.stream
        file_bytes = np.frombuffer(file_stream.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = detect_bubbles(MODEL_PATH, image)

        # Use hybrid OCR for multi-language support
        hybrid_ocr = Services.get_hybrid_ocr()
        manga_translator = Services.get_translator()

        logger.info(f"Source language: {source_lang} ({hybrid_ocr.get_engine_name(source_lang)})")
        logger.info(f"Target language: {target_lang}")

        ocr_images = []
        bubbles = []

        for result in results:
            x1, y1, x2, y2, score, class_id = result

            detected_image = image[int(y1):int(y2), int(x1):int(x2)]

            # Convert BGR to RGB for PIL
            rgb_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb_image)
            detected_image, cont = process_bubble(detected_image)
            ocr_images.append(im)
            bubbles.append((detected_image, cont))

        # Parallel OCR with hybrid engine
        max_workers = max(1, int(os.getenv("OCR_WORKERS", "4")))

        def ocr_one(img):
            return hybrid_ocr.read(img, lang=source_lang)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            texts = list(executor.map(ocr_one, ocr_images))

        translations = manga_translator.translate_batch(
            texts,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        for (detected_image, cont), text_translated in zip(bubbles, translations):
            add_text(detected_image, text_translated, f"fonts/{selected_font}i.ttf", cont)

        _, buffer = cv2.imencode(".png", image)
        image = buffer.tobytes()
        encoded_image = base64.b64encode(image).decode("utf-8")

        return render_template("translate.html", name=name, uploaded_image=encoded_image)

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
