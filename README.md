# Manga-Translator

Manga translation tool dengan AI-powered text detection, OCR, dan translation. Mendukung browser extension untuk realtime translation.

![Python](https://img.shields.io/badge/Python-3.10--3.11-blue)
![Flask](https://img.shields.io/badge/Flask-2.2+-green)
![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-yellow)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-purple)

## Features

- **Multi-mode Processing**: Realtime (<1s), Quality (2-3s), Premium (5-10s)
- **Advanced Detection**: YOLOv8/v11/v12 untuk speech bubble dan OSB (Outside Speech Bubble) text
- **Multi-language OCR**: Japanese (MangaOCR), Korean/Chinese/English (EasyOCR)
- **Smart Inpainting**: Canvas overlay, OpenCV, atau Flux AI
- **SAM2 Segmentation**: Precise bubble masking untuk quality/premium mode
- **Translation Cache**: LRU memory + SQLite persistent cache
- **GPU Support**: NVIDIA CUDA dan Intel GPU (OpenVINO)
- **Browser Extension**: Chrome/Edge extension untuk realtime translation

## Table of Contents

- [Quick Start](#quick-start)
- [Installation Options](#installation-options)
- [Configuration](#configuration)
- [API Usage](#api-usage)
- [Processing Modes](#processing-modes)
- [Browser Extension](#browser-extension)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

---

## Quick Start

### 1. Installation (Single Command)

```bash
# Clone repository
git clone https://github.com/user/manga-translator.git
cd manga-translator

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# FULL INSTALL (dependencies + all models)
python download_models.py
```

Script akan otomatis:
- Detect GPU (NVIDIA CUDA atau Intel)
- Install dependencies yang sesuai
- Download semua model yang diperlukan

### 2. Setup Environment

Copy `.env.example` ke `.env` dan isi API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Gemini API (untuk translation)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta/openai
GEMINI_MODEL=gemini-2.0-flash

# Server settings
HOST=0.0.0.0
PORT=5000
DEBUG=false

# Performance
OCR_WORKERS=4
```

### 3. Run Server

```bash
python app.py
```

Server akan berjalan di `http://localhost:5000`

---

## Installation Options

### Download Models Saja

```bash
# Realtime mode only (fastest)
python download_models.py realtime

# Quality mode (better accuracy)
python download_models.py quality

# Premium mode (best quality)
python download_models.py premium

# All models
python download_models.py all

# Check status
python download_models.py status
```

### Install Dependencies Saja

```bash
# Standard install
python download_models.py install

# CUDA install (untuk NVIDIA GPU)
pip install -r requirements-cuda.txt
```

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11 |
| RAM | 4 GB | 8 GB+ |
| CPU | Any x64 | Intel i5+ (untuk OpenVINO) |
| GPU | - | NVIDIA RTX / Intel Arc |
| Storage | 2 GB | 10 GB |

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Gemini API key | Required |
| `GEMINI_API_URL` | Gemini API endpoint | Google API |
| `GEMINI_MODEL` | Model name | `gemini-2.0-flash` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `5000` |
| `OCR_WORKERS` | Parallel OCR threads | `4` |
| `DEBUG` | Debug mode | `false` |

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

| Code | Language | Code | Language |
|------|----------|------|----------|
| `id` | Indonesian | `en` | English |
| `es` | Spanish | `fr` | French |
| `de` | German | `pt` | Portuguese |
| `ru` | Russian | `vi` | Vietnamese |
| `th` | Thai | `ar` | Arabic |
| `hi` | Hindi | `it` | Italian |
| `ms` | Malay | `tr` | Turkish |

---

## API Usage

### V1 API (Fast/Realtime)

```bash
curl -X POST http://localhost:5000/api/extension/translate \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/png;base64,...",
    "source_lang": "ja",
    "target_lang": "id"
  }'
```

Response:
```json
{
  "success": true,
  "regions": [
    {
      "bbox": [100, 50, 300, 150],
      "original": "こんにちは",
      "translated": "Halo",
      "font_size": 14
    }
  ],
  "timing_ms": 450
}
```

### V2 API (Advanced/Inpainting)

```bash
curl -X POST http://localhost:5000/api/extension/v2/translate \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/png;base64,...",
    "source_lang": "ja",
    "target_lang": "id",
    "mode": "quality",
    "detect_osb": true,
    "inpaint_method": "opencv"
  }'
```

Response:
```json
{
  "success": true,
  "image": "data:image/png;base64,...",
  "regions": [...],
  "meta": {
    "mode": "quality",
    "timing_ms": 2500,
    "cache_hit": false
  }
}
```

### Health Check

```bash
curl http://localhost:5000/health
```

---

## Processing Modes

| Mode | Speed | Features | Use Case |
|------|-------|----------|----------|
| **realtime** | <1s | Basic detection, canvas overlay | Browser extension, live reading |
| **quality** | 2-3s | SAM2 masks, OpenCV inpainting, OSB | Batch processing, better quality |
| **premium** | 5-10s | Flux AI inpainting, full pipeline | Final output, best quality |

### Mode Comparison

```
Realtime Mode:
- Detection: YOLOv8 speech bubble only
- Inpainting: Canvas overlay (solid color fill)
- OCR: Standard
- Speed: ~500ms

Quality Mode:
- Detection: YOLOv8 + OSB + Conjoined
- Inpainting: OpenCV TELEA/NS
- SAM2: Precise mask segmentation
- Speed: ~2500ms

Premium Mode:
- Detection: Full dual YOLO + SAM2
- Inpainting: Flux AI diffusion
- SAM2: High-res segmentation
- Speed: ~7000ms
```

---

## Browser Extension

### Build Extension

```bash
cd extension

# Install dependencies
npm install

# Build for production
npm run build

# Build for development (watch mode)
npm run dev
```

### Install Extension

1. Open Chrome/Edge
2. Go to `chrome://extensions` atau `edge://extensions`
3. Enable "Developer mode"
4. Click "Load unpacked"
5. Select `extension/dist` folder

### Extension Settings

| Setting | Options | Description |
|---------|---------|-------------|
| Mode | Realtime / Quality / Premium | Processing mode |
| Source Language | ja / ko / zh / en | Manga language |
| Target Language | id / en / etc | Translation target |
| Detect OSB | On / Off | Outside-bubble text |
| Inpaint Method | Auto / Canvas / OpenCV / Flux | Text removal method |
| Use Advanced | On / Off | SAM2 + dual YOLO |

---

## Docker Deployment

### Quick Start with Docker

```bash
# Build image
docker build -t manga-translator .

# Run container
docker run -d \
  --name manga-translator \
  -p 5000:5000 \
  -e GEMINI_API_KEY=your_key \
  -v manga-models:/app/model \
  manga-translator
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  manga-translator:
    build: .
    ports:
      - "5000:5000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GEMINI_API_URL=${GEMINI_API_URL:-https://generativelanguage.googleapis.com/v1beta/openai}
      - GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.0-flash}
      - HOST=0.0.0.0
      - PORT=5000
      - OCR_WORKERS=4
    volumes:
      - manga-models:/app/model
      - manga-cache:/app/cache
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  manga-models:
  manga-cache:
```

Run:
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download models
RUN python download_models.py realtime

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["python", "app.py"]
```

### Docker with NVIDIA GPU

Create `docker-compose.gpu.yml`:

```yaml
version: '3.8'

services:
  manga-translator:
    build: .
    ports:
      - "5000:5000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OCR_WORKERS=4
    volumes:
      - manga-models:/app/model
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  manga-models:
```

Run:
```bash
docker-compose -f docker-compose.gpu.yml up -d
```

### Docker Multi-Stage Build (Optimized)

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY . .

# Download models
RUN python download_models.py realtime

EXPOSE 5000
CMD ["python", "app.py"]
```

---

## Production Deployment

### Using Gunicorn (Linux)

```bash
# Install gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# With timeout for long requests
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
```

### Using Waitress (Windows/Linux)

```bash
# Install waitress
pip install waitress

# Run server
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/manga-translator
server {
    listen 80;
    server_name manga.yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name manga.yourdomain.com;

    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/manga.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/manga.yourdomain.com/privkey.pem;

    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # For large images
        client_max_body_size 50M;
        proxy_read_timeout 120s;
        proxy_connect_timeout 60s;
    }

    # WebSocket support (if needed)
    location /ws {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/manga-translator /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Systemd Service (Linux)

Create `/etc/systemd/system/manga-translator.service`:

```ini
[Unit]
Description=Manga Translator API
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/manga-translator
Environment="PATH=/opt/manga-translator/.venv/bin"
EnvironmentFile=/opt/manga-translator/.env
ExecStart=/opt/manga-translator/.venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 --timeout 120 app:app
Restart=always
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/manga-translator

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable manga-translator
sudo systemctl start manga-translator
sudo systemctl status manga-translator
```

### PM2 (Node.js Process Manager)

```bash
# Install PM2
npm install -g pm2

# Create ecosystem file
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [{
    name: 'manga-translator',
    script: 'gunicorn',
    args: '-w 4 -b 0.0.0.0:5000 app:app',
    interpreter: '/opt/manga-translator/.venv/bin/python',
    cwd: '/opt/manga-translator',
    env: {
      GEMINI_API_KEY: 'your_key'
    }
  }]
}
EOF

# Start
pm2 start ecosystem.config.js

# Save and enable startup
pm2 save
pm2 startup
```

---

## Cloud Deployment

### Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

Create `railway.json`:
```json
{
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "startCommand": "python app.py",
    "healthcheckPath": "/health",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

### Render

1. Connect GitHub repository
2. Create new Web Service
3. Configure:
   - **Build Command**: `pip install -r requirements.txt && python download_models.py realtime`
   - **Start Command**: `gunicorn -w 2 -b 0.0.0.0:$PORT app:app`
4. Set environment variables
5. Deploy

### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Launch app
fly launch

# Deploy
fly deploy
```

Create `fly.toml`:
```toml
app = "manga-translator"
primary_region = "sin"

[build]
  dockerfile = "Dockerfile"

[env]
  PORT = "8080"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  cpu_kind = "shared"
  cpus = 2
  memory_mb = 2048
```

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/manga-translator

# Deploy
gcloud run deploy manga-translator \
  --image gcr.io/PROJECT_ID/manga-translator \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 120s \
  --set-env-vars "GEMINI_API_KEY=your_key"
```

### AWS ECS

```bash
# Create ECR repository
aws ecr create-repository --repository-name manga-translator

# Build and push
docker build -t manga-translator .
docker tag manga-translator:latest AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/manga-translator:latest
aws ecr get-login-password | docker login --username AWS --password-stdin AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker push AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/manga-translator:latest
```

---

## Project Structure

```
manga-translator/
├── app.py                  # Flask application entry point
├── api_extension.py        # Extension API routes (V1 & V2)
├── pipeline.py             # Main translation pipeline
├── services.py             # Singleton model services
├── config.py               # Configuration dataclasses
├── cache.py                # Translation & image caching
├── detect_bubbles.py       # YOLO bubble detection
├── inpainting.py           # Text removal (canvas/opencv/flux)
├── ocr_hybrid.py           # Multi-language OCR
├── advanced_detection.py   # SAM2 + dual YOLO detection
├── advanced_cleaning.py    # Advanced text cleaning
├── download_models.py      # Model downloader
├── process_bubble.py       # Bubble processing
├── add_text.py             # Text rendering
├── bubble_split.py         # Merged bubble splitting
├── osb_detection.py        # Outside-speech-bubble detection
├── export_openvino.py      # OpenVINO export script
│
├── model/                  # Downloaded models
│   ├── yolov8m_seg-speech-bubble.pt
│   ├── animetext_yolov12x.pt
│   ├── comic-speech-bubble-detector-yolov8m.pt
│   ├── manga109_panel_yolov11.pt
│   └── sam/
│
├── translator/             # Translation module
│   └── translator.py
│
├── extension/              # Browser extension
│   ├── src/
│   │   ├── content.ts
│   │   ├── background.ts
│   │   ├── popup.ts
│   │   ├── cache/
│   │   │   └── TranslatedImageCache.ts
│   │   └── replacer/
│   │       └── ImageReplacer.ts
│   ├── public/
│   │   ├── popup.html
│   │   ├── popup.css
│   │   └── manifest.json
│   ├── dist/               # Built extension
│   └── package.json
│
├── fonts/                  # Font files
│   └── animeace_i.ttf
│
├── templates/              # Flask HTML templates
├── static/                 # Static assets
│
├── requirements.txt        # Python dependencies
├── requirements-cuda.txt   # CUDA dependencies
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Troubleshooting

### Model tidak terdownload

```bash
# Check status
python download_models.py status

# Force re-download
python download_models.py all
```

### CUDA tidak terdeteksi

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Intel GPU tidak terdeteksi

```bash
# Install OpenVINO
pip install openvino

# Check Intel GPU
python -c "from openvino import Core; print(Core().available_devices)"
```

### Memory error saat load model

```python
# Unload Flux model setelah selesai
from services import Services
Services.unload_flux()
```

### OCR lambat

```bash
# Gunakan CUDA untuk EasyOCR (NVIDIA only)
pip install easyocr[gpu]

# Atau kurangi workers
export OCR_WORKERS=2
```

### Extension tidak connect

1. Pastikan server berjalan di `http://localhost:5000`
2. Check health endpoint: `curl http://localhost:5000/health`
3. Check CORS settings di `app.py`
4. Reload extension di browser

### numpy version conflict

```bash
pip install 'numpy>=1.24.2,<2.0'
```

### Translation quality issues

```env
# Gunakan model yang lebih baik
GEMINI_MODEL=gemini-1.5-pro

# Atau model yang lebih cepat
GEMINI_MODEL=gemini-2.0-flash
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/status` | Model status |
| POST | `/api/extension/translate` | V1 translate (fast) |
| POST | `/api/extension/v2/translate` | V2 translate (advanced) |
| POST | `/api/extension/v2/cache/clear` | Clear V2 cache |
| GET | `/api/extension/v2/cache/stats` | V2 cache statistics |
| POST | `/v1/translate_viewport` | Translate single image |
| POST | `/v1/batch_detect` | Batch bubble detection |
| POST | `/v1/batch_translate` | Batch translation |

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | string | required | Base64 encoded image |
| `source_lang` | string | "ja" | Source language (ja/ko/zh/en) |
| `target_lang` | string | "id" | Target language |
| `mode` | string | "realtime" | Processing mode |
| `detect_osb` | boolean | true | Detect outside-bubble text |
| `inpaint_method` | string | "auto" | Inpainting method |
| `use_advanced` | boolean | true | Use SAM2 + dual YOLO |

### Response Format

```json
{
  "success": true,
  "regions": [
    {
      "id": "r0",
      "bbox": [x1, y1, x2, y2],
      "original": "原文",
      "translated": "Translation",
      "confidence": 0.95,
      "font_size": 14,
      "mask_type": "ellipse"
    }
  ],
  "meta": {
    "mode": "quality",
    "timing_ms": 2500,
    "detect_ms": 120,
    "ocr_ms": 450,
    "translate_ms": 200,
    "cache_hit": false
  },
  "image": "data:image/png;base64,..." // V2 only
}
```

---

## Performance Tips

1. **Use OpenVINO INT8** for 2-3x faster detection on Intel CPUs:
   ```bash
   python export_openvino.py --int8
   ```

2. **Use CUDA** for NVIDIA GPUs (5-10x faster):
   ```bash
   pip install -r requirements-cuda.txt
   ```

3. **Increase OCR workers** for multi-core CPUs:
   ```env
   OCR_WORKERS=8
   ```

4. **Enable translation cache** - automatically caches repeated text

5. **Use realtime mode** for browser extension (fastest)

6. **Unload Flux** after batch processing to free memory:
   ```python
   Services.unload_flux()
   ```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Credits

- [MangaOCR](https://github.com/kha-white/manga-ocr) - Japanese OCR
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Multi-language OCR
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO models
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Segmentation
- [Flux](https://huggingface.co/black-forest-labs) - AI Inpainting
- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel optimization
- [Gemini API](https://ai.google.dev/) - Translation engine
