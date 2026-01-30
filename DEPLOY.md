# Manga Translator - Deployment Guide

## Quick Start

### 1. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask app (development)
python app.py

# Run Gradio app (development)
python app_optimized.py
```

### 2. Production with Gunicorn
```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn (4 workers)
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Run with more workers for high traffic
gunicorn -w 8 -b 0.0.0.0:5000 --timeout 120 app:app
```

### 3. Docker Deployment
```bash
# Build image
docker build -t manga-translator .

# Run container
docker run -p 5000:5000 manga-translator

# Run with GPU (NVIDIA)
docker run --gpus all -p 5000:5000 manga-translator
```

---

## Deployment Options

### Option A: Standalone Server

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Preload models (optional, reduces first request time):**
   ```bash
   python -c "from services import Services; Services.preload_all()"
   ```

3. **Run production server:**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
   ```

### Option B: Docker

See `Dockerfile` for container build.

### Option C: Systemd Service (Linux)

1. Create service file `/etc/systemd/system/manga-translator.service`:
   ```ini
   [Unit]
   Description=Manga Translator API
   After=network.target

   [Service]
   User=www-data
   WorkingDirectory=/opt/manga-translator
   ExecStart=/opt/manga-translator/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

2. Enable and start:
   ```bash
   sudo systemctl enable manga-translator
   sudo systemctl start manga-translator
   ```

### Option D: Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name manga-translator.example.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
        client_max_body_size 50M;
    }
}
```

---

## API Endpoints

### Health Check
```
GET /health
```

### V1 API (Browser Extension)
```
POST /v1/translate
Content-Type: application/json

{
    "image": "base64_encoded_image",
    "target_lang": "id",
    "source_lang": "ja"
}
```

### V2 API (Advanced)
```
POST /v2/translate
Content-Type: application/json

{
    "image": "base64_encoded_image",
    "target_lang": "id",
    "source_lang": "ja",
    "mode": "quality",
    "detect_osb": true,
    "inpaint_method": "opencv"
}
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask secret key | `secret_key` |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `OCR_WORKERS` | Number of OCR workers | `4` |
| `PORT` | Server port | `5000` |

---

## Performance Tuning

### For CPU (Intel i7)
```bash
# Set thread count
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Use OpenVINO optimized model
python export_openvino.py --int8
```

### For GPU (NVIDIA)
```bash
# Ensure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Models will automatically use GPU
```

### For Intel GPU
```bash
# Install OpenVINO
pip install openvino>=2024.0.0

# Models will automatically use Intel GPU via OpenVINO
```

---

## Monitoring

### Check service status
```bash
curl http://localhost:5000/health
```

### View logs
```bash
# Systemd
journalctl -u manga-translator -f

# Docker
docker logs -f manga-translator
```
