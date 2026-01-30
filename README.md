# Manga-Translator

Manga translation tool dengan AI-powered text detection, OCR, dan translation.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-2.2+-green)
![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-yellow)

## Features

- **Speech Bubble Detection**: YOLOv8 untuk deteksi bubble
- **OSB Detection**: Deteksi teks di luar bubble (SFX, narasi, judul)
- **Multi-language OCR**: Japanese (MangaOCR), Korean/Chinese/English (EasyOCR)
- **Inpainting**: OpenCV atau Canvas untuk menghapus teks original
- **SAM2 Segmentation**: Precise bubble masking (optional)
- **Translation Cache**: Cache untuk mempercepat terjemahan berulang
- **Browser Extension**: Chrome/Edge extension dengan interactive settings

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/user/manga-translator.git
cd manga-translator

# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies + download models
python download_models.py
```

### 2. Setup Environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

### 3. Run Server

```bash
python app.py
```

Server berjalan di `http://localhost:5000`

---

## API Usage

### Unified API (`/api/translate`)

Single endpoint dengan feature toggles:

```bash
curl -X POST http://localhost:5000/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "image_b64": "data:image/png;base64,...",
    "source_lang": "ja",
    "target_lang": "id",
    "detect_osb": true,
    "use_inpainting": true,
    "inpaint_method": "opencv",
    "use_sam": false,
    "use_advanced": false,
    "render_text": true
  }'
```

### Feature Toggles

| Feature | Default | Description |
|---------|---------|-------------|
| `detect_osb` | `true` | Deteksi SFX, narasi di luar bubble |
| `use_inpainting` | `true` | Hapus teks original |
| `inpaint_method` | `opencv` | `opencv` (clean) atau `canvas` (fast) |
| `use_sam` | `false` | SAM2 untuk mask presisi (slower) |
| `use_advanced` | `false` | Dual-YOLO detection (slower) |
| `render_text` | `true` | Render teks terjemahan ke gambar |

### Response

```json
{
  "success": true,
  "regions": [
    {
      "id": "r0",
      "bbox": [100, 50, 200, 100],
      "type": "bubble",
      "src_text": "こんにちは",
      "tgt_text": "Halo",
      "confidence": 0.95
    }
  ],
  "image_b64": "iVBORw0KGgo...",
  "meta": {
    "bubble_count": 5,
    "osb_count": 2,
    "timings": {
      "detection_ms": 45,
      "ocr_ms": 200,
      "translate_ms": 150,
      "inpaint_ms": 80
    }
  }
}
```

### Health Check

```bash
curl http://localhost:5000/api/health
```

---

## Browser Extension

### Build

```bash
cd extension
npm install
npm run build
```

### Install

1. Buka `chrome://extensions`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select folder `extension/dist`

### Settings

Extension popup menyediakan checkbox untuk setiap feature:

| Setting | Description |
|---------|-------------|
| OSB Detection | Deteksi teks di luar bubble |
| Inpainting | Hapus teks original |
| Inpaint Method | OpenCV (clean) / Canvas (fast) |
| SAM2 Masks | Segmentasi bubble presisi |
| Advanced Detection | Dual-YOLO untuk bubble conjoined |
| Render Text | Tampilkan teks terjemahan |

---

## Supported Languages

**Source (OCR):**
| Code | Language | Engine |
|------|----------|--------|
| `ja` | Japanese | MangaOCR |
| `ko` | Korean | EasyOCR |
| `zh` | Chinese (Simplified) | EasyOCR |
| `zh-tw` | Chinese (Traditional) | EasyOCR |
| `en` | English | EasyOCR |

**Target (Translation):**
`id`, `en`, `es`, `fr`, `de`, `pt`, `ru`, `vi`, `th`, `ar`, `hi`, `it`, `ms`, `tr`

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Gemini API key | Required |
| `GEMINI_MODEL` | Model name | `gemini-2.0-flash` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `5000` |
| `OCR_WORKERS` | Parallel OCR threads | `4` |

---

## Download Models

```bash
# Check status
python download_models.py status

# Download all models
python download_models.py all

# Download specific
python download_models.py yolo
python download_models.py manga_ocr
python download_models.py sam
```

---

## Docker

```bash
# Build
docker build -t manga-translator .

# Run
docker run -d \
  --name manga-translator \
  -p 5000:5000 \
  -e GEMINI_API_KEY=your_key \
  manga-translator
```

---

## Project Structure

```
manga-translator/
├── app.py                  # Flask entry point
├── api_unified.py          # Unified API (/api/translate)
├── api_extension.py        # Legacy API (V1/V2)
├── services.py             # Model services
├── detect_bubbles.py       # YOLO detection
├── ocr_hybrid.py           # Multi-language OCR
├── osb_detection.py        # Outside-bubble detection
├── inpainting.py           # Text removal
├── translator/             # Translation module
├── extension/              # Browser extension
│   ├── src/                # TypeScript source
│   ├── public/             # Static files
│   └── dist/               # Built extension
├── model/                  # Downloaded models
└── requirements.txt
```

---

## Troubleshooting

### Extension tidak connect
1. Pastikan server berjalan di `http://localhost:5000`
2. Check: `curl http://localhost:5000/api/health`
3. Reload extension di browser

### OCR lambat
```bash
# Kurangi workers jika memory terbatas
export OCR_WORKERS=2
```

### numpy conflict
```bash
pip install 'numpy>=1.24.2,<2.0'
```

---

## License

MIT License

## Credits

- [MangaOCR](https://github.com/kha-white/manga-ocr)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Gemini API](https://ai.google.dev/)
