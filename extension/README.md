# Manga Translator Chrome Extension

Real-time manga translation extension that works with any manga website. Translates speech bubbles directly on the page using image replacement rendering.

![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-blue)
![Manifest V3](https://img.shields.io/badge/Manifest-V3-green)
![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)

## Features

- **Real-time Translation** — Translate manga pages while browsing
- **Image Replacement** — Renders translated text directly onto images (not DOM overlay)
- **Toggle View** — Switch between original and translated images
- **Multi-Language Support** — Japanese, Korean, Chinese to 15+ target languages
- **Batch Processing** — Processes multiple images in parallel
- **Persistent Settings** — Remembers your language preferences

## Requirements

- Chrome/Chromium browser (version 88+)
- [Manga Translator Backend](../Manga-Translator) running locally

## Installation

### From Source

1. **Install dependencies:**

```bash
npm install
```

2. **Build the extension:**

```bash
npm run build
```

3. **Load in Chrome:**
   - Open `chrome://extensions/`
   - Enable **Developer mode**
   - Click **Load unpacked**
   - Select the `dist/` folder

### Development Mode

Watch for changes and rebuild automatically:

```bash
npm run dev
```

## Usage

### 1. Start Backend Server

```bash
cd ../Manga-Translator
python app.py
```

Backend runs at `http://127.0.0.1:5000`

### 2. Configure Extension

Click the extension icon and configure:

| Setting | Description | Default |
|---------|-------------|---------|
| Source Language | Language of the manga | Japanese |
| Target Language | Your preferred language | Indonesian |
| Backend URL | Backend server address | `http://127.0.0.1:5000` |

### 3. Translate

1. Navigate to any manga website
2. Click **Translate Page**
3. Wait for translation to complete
4. Use **Toggle Original** to compare

## Project Structure

```
manga-translator-extension/
├── src/
│   ├── background.ts          # Service worker (API calls)
│   ├── content.ts             # Content script (DOM manipulation)
│   ├── popup.ts               # Popup UI logic
│   ├── cache/
│   │   └── TranslatedImageCache.ts  # Canvas rendering + caching
│   └── replacer/
│       └── ImageReplacer.ts   # Image src replacement
├── public/
│   ├── manifest.json          # Chrome extension manifest
│   ├── popup.html             # Popup UI
│   ├── popup.css              # Popup styles
│   ├── content.css            # Content script styles
│   └── icons/                 # Extension icons
├── dist/                      # Built extension (load this in Chrome)
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Content Script                            │
│                                                              │
│  1. Find manga images on page                                │
│  2. Convert to base64                                        │
│  3. Send to background worker                                │
│                                                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Background Worker                            │
│                                                              │
│  4. POST to backend /v1/translate_viewport                   │
│  5. Receive translation regions                              │
│                                                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              TranslatedImageCache                            │
│                                                              │
│  6. Draw original image on canvas                            │
│  7. Render white masks + translated text                     │
│  8. Convert canvas to blob URL                               │
│                                                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  ImageReplacer                               │
│                                                              │
│  9. Replace image.src with blob URL                          │
│ 10. Store original src for toggle                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## API Communication

### Translate Image

```typescript
chrome.runtime.sendMessage({
  action: 'translateImage',
  data: {
    imageHash: 'abc123',
    imageBase64: 'data:image/png;base64,...',
    sourceLang: 'ja',
    targetLang: 'id'
  }
});
```

### Response

```typescript
{
  success: true,
  regions: [
    {
      id: 'r0',
      bbox: [100, 50, 200, 80],
      src_text: 'こんにちは',
      tgt_text: 'Halo',
      render: { font_size: 14, align: 'center' }
    }
  ],
  meta: {
    detect_ms: 120,
    ocr_ms: 450,
    translate_ms: 200
  }
}
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Click extension icon | Open popup |
| Alt+T | Translate page (configurable) |
| Alt+O | Toggle original/translated |

## Troubleshooting

### Extension can't connect to backend

1. Check backend is running: `curl http://127.0.0.1:5000/health`
2. Verify Backend URL in extension settings
3. Check browser console for CORS errors

### Images not detected

The extension filters images by:
- Minimum size: 300x400 pixels
- Aspect ratio: 0.3 - 1.5 (portrait/square)
- Must be fully loaded

### Translation quality

Translation quality depends on:
- OCR accuracy (try different source languages)
- Gemini model (configure in backend `.env`)

## Development

### Build Commands

```bash
npm run dev      # Watch mode
npm run build    # Production build
npm run preview  # Preview build
```

### TypeScript Configuration

- Target: ES2020
- Module: ESNext
- Strict mode enabled

### Adding New Features

1. Modify source files in `src/`
2. Run `npm run build`
3. Reload extension in Chrome

## License

MIT License - see main project for details.
