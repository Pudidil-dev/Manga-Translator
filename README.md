# Manga Translator

A powerful manga translation tool that automatically detects speech bubbles, extracts text using OCR, and translates it to your preferred language. Includes a Chrome extension for real-time translation while browsing manga websites.

![Python](https://img.shields.io/badge/Python-3.8--3.11-blue)
![Flask](https://img.shields.io/badge/Flask-2.2+-green)
![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-yellow)
![License](https://img.shields.io/badge/License-MIT-purple)

## Features

- **Bubble Detection** — YOLOv8 with OpenVINO INT8 optimization for Intel CPUs
- **Multi-Language OCR** — MangaOCR (Japanese) + EasyOCR (Korean, Chinese, English)
- **AI Translation** — Gemini API for natural, context-aware translations
- **Chrome Extension** — Real-time translation with image replacement rendering
- **Batch Processing** — Parallel OCR and translation for faster processing
- **Translation Cache** — In-memory LRU + SQLite for persistent caching

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [Command Line](#command-line)
  - [Chrome Extension](#chrome-extension)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Requirements

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.10 - 3.11 |
| RAM | 4 GB | 8 GB+ |
| CPU | Any x64 | Intel i5+ (for OpenVINO) |
| Storage | 2 GB | 5 GB |

### API Requirements

- **Gemini API Key** — Get from [Google AI Studio](https://aistudio.google.com/apikey)
- Or use a local Gemini proxy (e.g., `localhost:8317`)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/Manga-Translator.git
cd Manga-Translator
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```env
# Gemini API Configuration
GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta/models
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash

# Or use local proxy
# GEMINI_API_URL=http://localhost:8317/
# GEMINI_API_KEY=your_proxy_key
# GEMINI_MODEL=gemini-2.5-flash-lite

# Performance Settings
OCR_WORKERS=4
```

### 5. Download/Export OpenVINO Model (Optional but Recommended)

For 2-3x faster inference on Intel CPUs:

```bash
python export_openvino.py --int8
```

This creates `model/model_int8_openvino_model/` for optimized inference.

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_URL` | Gemini API endpoint | Google API |
| `GEMINI_API_KEY` | Your API key | Required |
| `GEMINI_MODEL` | Model name | `gemini-2.0-flash` |
| `OCR_WORKERS` | Parallel OCR threads | `4` |
| `SECRET_KEY` | Flask secret key | `secret_key` |

### Supported Languages

**Source Languages (OCR):**
| Code | Language | OCR Engine |
|------|----------|------------|
| `ja` | Japanese | MangaOCR |
| `ko` | Korean | EasyOCR |
| `zh` | Chinese (Simplified) | EasyOCR |
| `zh-tw` | Chinese (Traditional) | EasyOCR |
| `en` | English | EasyOCR |

**Target Languages (Translation):**
| Code | Language |
|------|----------|
| `id` | Indonesian |
| `en` | English |
| `es` | Spanish |
| `fr` | French |
| `de` | German |
| `pt` | Portuguese |
| `ru` | Russian |
| `vi` | Vietnamese |
| `th` | Thai |
| `ar` | Arabic |
| `hi` | Hindi |
| `it` | Italian |
| `nl` | Dutch |
| `pl` | Polish |
| `tr` | Turkish |
| `ms` | Malay |

---

## Usage

### Web Interface

1. **Start the server:**

```bash
python app.py
```

2. **Open browser:** http://localhost:5000

3. **Upload manga image** and select languages

4. **Download** the translated image

### Command Line

```bash
python main.py \
  --model-path model/model.pt \
  --image-path input.png \
  --source-lang ja \
  --target-lang en \
  --save-path output/
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-path` | YOLO model path | `model/model.pt` |
| `--image-path` | Input image | Required |
| `--font-path` | Font file | `fonts/animeace_i.ttf` |
| `--source-lang` | Source language | `ja` |
| `--target-lang` | Target language | `en` |
| `--save-path` | Output directory | Current dir |

### Chrome Extension

#### Installation

1. **Build the extension** (if not already built):

```bash
cd ../manga-translator-extension
npm install
npm run build
```

2. **Load in Chrome:**
   - Open `chrome://extensions/`
   - Enable **Developer mode** (top-right toggle)
   - Click **Load unpacked**
   - Select `manga-translator-extension/dist/` folder

3. **Start backend server:**

```bash
cd ../Manga-Translator
python app.py
```

#### Using the Extension

1. Navigate to any manga website (e.g., MangaDex, RawKuma, etc.)

2. Click the extension icon in Chrome toolbar

3. Configure settings:
   - **Source Language:** Language of the manga
   - **Target Language:** Your preferred language
   - **Backend URL:** `http://127.0.0.1:5000` (default)

4. Click **Translate Page**

5. Use **Toggle Original** to switch between original and translated

6. Right-click images to save translated versions directly

---

## API Reference

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "ok": true,
  "model_loaded": true,
  "yolo_format": "openvino_int8",
  "cache_stats": {
    "memory_items": 42,
    "db_items": 0,
    "max_memory": 2000
  }
}
```

### Translate Single Image

```http
POST /v1/translate_viewport
Content-Type: application/json

{
  "image_b64": "data:image/png;base64,...",
  "source_lang": "ja",
  "target_lang": "id"
}
```

**Response:**
```json
{
  "regions": [
    {
      "id": "r0",
      "bbox": [100, 50, 200, 80],
      "src_text": "こんにちは",
      "tgt_text": "Halo",
      "confidence": 0.95,
      "render": {
        "font_size": 14,
        "align": "center"
      }
    }
  ],
  "meta": {
    "detect_ms": 120,
    "ocr_ms": 450,
    "translate_ms": 200,
    "cached_hits": 0
  }
}
```

### Batch Detection

```http
POST /v1/batch_detect
Content-Type: application/json

{
  "images": [
    {"image_id": "img1", "image_b64": "..."},
    {"image_id": "img2", "image_b64": "..."}
  ],
  "source_lang": "ja"
}
```

### Batch Translation

```http
POST /v1/batch_translate
Content-Type: application/json

{
  "bubbles": [...],
  "source_lang": "ja",
  "target_lang": "id"
}
```

### Cache Management

```http
GET /v1/cache/stats
POST /v1/cache/clear
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Chrome Extension                             │
│  ┌──────────┐    ┌─────────────────┐    ┌───────────────────┐   │
│  │ Popup UI │───▶│ Background      │───▶│ Content Script    │   │
│  │          │    │ Service Worker  │    │ - Image Detection │   │
│  │ Settings │    │ - API Calls     │    │ - Canvas Render   │   │
│  └──────────┘    └────────┬────────┘    └───────────────────┘   │
└───────────────────────────┼─────────────────────────────────────┘
                            │ HTTP/REST
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (Flask)                              │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ YOLO        │  │ HybridOCR   │  │ Gemini      │              │
│  │ OpenVINO    │  │ MangaOCR +  │  │ Translation │              │
│  │ INT8        │  │ EasyOCR     │  │ API         │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│         │                │                │                      │
│         └────────────────┴────────────────┘                      │
│                          │                                       │
│                   ┌──────┴──────┐                                │
│                   │ Translation │                                │
│                   │ Cache (LRU) │                                │
│                   └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
Manga-Translator/
├── app.py                 # Flask web application
├── pipeline.py            # Translation pipeline
├── services.py            # Singleton model services
├── cache.py               # Translation cache
├── api_extension.py       # Extension API routes
├── detect_bubbles.py      # YOLO bubble detection
├── ocr_hybrid.py          # Multi-language OCR
├── process_bubble.py      # Bubble whitening
├── add_text.py            # Text rendering
├── translator/
│   └── translator.py      # Gemini translation
├── model/
│   ├── model.pt           # YOLO PyTorch model
│   └── model_int8_openvino_model/  # Optimized model
├── fonts/
│   └── animeace_i.ttf     # Default font
├── templates/             # Flask HTML templates
├── static/                # Static assets
├── .env                   # Configuration
└── requirements.txt       # Python dependencies

manga-translator-extension/
├── src/
│   ├── background.ts      # Service worker
│   ├── content.ts         # Content script
│   ├── popup.ts           # Popup UI logic
│   ├── cache/
│   │   └── TranslatedImageCache.ts
│   └── replacer/
│       └── ImageReplacer.ts
├── public/
│   ├── manifest.json      # Chrome manifest v3
│   ├── popup.html
│   ├── popup.css
│   └── icons/
├── dist/                  # Built extension
├── package.json
├── tsconfig.json
└── vite.config.ts
```

---

## Troubleshooting

### Common Issues

#### 1. `numpy` version conflict

```
RuntimeError: Numpy is not available
```

**Solution:**
```bash
pip install 'numpy>=1.24.2,<2.0'
```

#### 2. YOLO model not loading

```
Unable to automatically guess model task
```

**Solution:** This is a warning, not an error. The model will still work.

#### 3. MangaOCR slow first load

First request takes longer (~10s) because models are downloaded. Subsequent requests are fast.

#### 4. Extension can't connect to backend

**Check:**
1. Backend is running: `python app.py`
2. CORS is enabled (check `/health` endpoint)
3. Backend URL in extension matches (default: `http://127.0.0.1:5000`)

#### 5. Translation quality issues

Try adjusting the Gemini model:
```env
GEMINI_MODEL=gemini-2.0-flash      # Faster, good quality
GEMINI_MODEL=gemini-1.5-pro        # Slower, better quality
```

### Performance Tips

1. **Use OpenVINO INT8** for 2-3x faster detection:
   ```bash
   python export_openvino.py --int8
   ```

2. **Increase OCR workers** for multi-core CPUs:
   ```env
   OCR_WORKERS=8
   ```

3. **Enable translation cache** to avoid re-translating same text

4. **Use SSD** for faster model loading

---

## Datasets

Training datasets used for bubble detection:

- [Manga Bubble Dataset 1](https://universe.roboflow.com/luciano-bastos-nunes/mangas-bubble)
- [Manga Bubble Dataset 2](https://universe.roboflow.com/sheepymeh/manga-vd5mb/)

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [MangaOCR](https://github.com/kha-white/manga-ocr) — Japanese manga OCR
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — Multi-language OCR
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection
- [OpenVINO](https://github.com/openvinotoolkit/openvino) — Intel CPU optimization
- [Gemini API](https://ai.google.dev/) — Translation engine
