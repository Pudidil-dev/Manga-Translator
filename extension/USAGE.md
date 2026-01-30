# Extension Usage Guide - Unified API

## Overview

Extension menggunakan single Unified API (`/api/translate`) dengan feature toggles interaktif.
User memilih fitur yang diinginkan via checkbox di popup.

## Feature Toggles

| Feature | Default | Deskripsi |
|---------|---------|-----------|
| `detectOsb` | `true` | Deteksi SFX, narasi, judul di luar bubble |
| `useInpainting` | `true` | Hapus teks original sebelum render |
| `inpaintMethod` | `opencv` | Metode inpainting: `opencv` (fast) atau `canvas` (fastest) |
| `useSam` | `false` | SAM2 untuk segmentasi bubble presisi (slower) |
| `useAdvanced` | `false` | Dual-YOLO untuk deteksi bubble conjoined (slower) |
| `renderText` | `true` | Render teks terjemahan ke gambar |

## Cara Pakai dari Content Script

### Translate Image

```javascript
chrome.runtime.sendMessage({
  action: 'translate',
  data: {
    imageHash: 'hash123',
    imageBase64: 'data:image/png;base64,...',
    // Feature toggles (optional - uses settings if not provided)
    sourceLang: 'ja',
    targetLang: 'id',
    detectOsb: true,
    useInpainting: true,
    inpaintMethod: 'opencv',
    useSam: false,
    useAdvanced: false,
    renderText: true
  }
}, (response) => {
  if (response.success) {
    console.log('Regions:', response.regions);

    // Processed image with inpainting + text
    if (response.image_b64) {
      const img = new Image();
      img.src = 'data:image/png;base64,' + response.image_b64;
    }

    console.log('Meta:', response.meta);
  }
});
```

### Get/Save Settings

```javascript
// Get settings
chrome.runtime.sendMessage({ action: 'getSettings' }, (response) => {
  console.log('Settings:', response.settings);
});

// Save settings
chrome.runtime.sendMessage({
  action: 'saveSettings',
  data: {
    sourceLang: 'ja',
    targetLang: 'id',
    detectOsb: true,
    useInpainting: true,
    inpaintMethod: 'opencv',
    useSam: false,
    useAdvanced: false,
    renderText: true
  }
});
```

### Check Health

```javascript
chrome.runtime.sendMessage({ action: 'checkHealth' }, (response) => {
  if (response.success) {
    console.log('Backend OK:', response.health.ok);
    console.log('Models:', response.health.models);
  }
});
```

## Response Format

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
      "confidence": 0.95,
      "render": {
        "font_size": 14,
        "align": "center"
      }
    },
    {
      "id": "r1",
      "bbox": [300, 200, 80, 40],
      "type": "osb",
      "src_text": "ドン！",
      "tgt_text": "DON!",
      "confidence": 0.8
    }
  ],
  "image_b64": "iVBORw0KGgo...",
  "meta": {
    "image_hash": "abc123def",
    "bubble_count": 3,
    "osb_count": 2,
    "total_regions": 5,
    "features": {
      "sam": false,
      "advanced": false,
      "osb": true,
      "inpainting": true,
      "inpaint_method": "opencv",
      "render_text": true
    },
    "timings": {
      "detection_ms": 45,
      "osb_ms": 30,
      "ocr_translate_ms": 350,
      "inpaint_ms": 80,
      "render_ms": 20
    }
  }
}
```

## Feature Combinations

| Use Case | Recommended Settings |
|----------|---------------------|
| **Fast Preview** | `useInpainting=false, renderText=false` |
| **Standard** | Default settings |
| **High Quality** | `useSam=true, useInpainting=true` |
| **Full Detection** | `useAdvanced=true, detectOsb=true` |

## Rebuild Extension

Setelah update TypeScript:

```bash
cd extension
npm install
npm run build
```

Lalu reload extension di Chrome:
1. Buka `chrome://extensions/`
2. Klik "Reload" pada extension
