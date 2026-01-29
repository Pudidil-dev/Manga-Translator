/**
 * Cache for rendered translated images.
 * Uses Canvas API to render text onto images and stores as Blob.
 */

export interface Region {
  id: string;
  bbox: [number, number, number, number]; // [x, y, w, h]
  polygon: number[][];
  mask_type: 'ellipse' | 'rect' | 'polygon' | 'text_only';
  src_text: string;
  tgt_text: string;
  confidence: number;
  render: {
    font_size: number;
    font_family: string;
    align: string;
    line_height: number;
    padding: number;
  };
}

interface CachedImage {
  originalSrc: string;
  translatedBlob: Blob;
  translatedUrl: string;
  regions: Region[];
  width: number;
  height: number;
  timestamp: number;
}

export class TranslatedImageCache {
  private memoryCache: Map<string, CachedImage> = new Map();
  private maxMemoryItems: number = 50;

  /**
   * Render image with translations and cache the result.
   * @param image - Clean HTMLImageElement (loaded from base64, no CORS issues)
   * @param regions - Translation regions from backend
   * @param hash - Unique hash for caching
   */
  async renderAndCache(
    image: HTMLImageElement,
    regions: Region[],
    hash: string
  ): Promise<string> {
    console.log('[MangaTranslate Cache] Rendering image:', hash, 'with', regions.length, 'regions');

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      throw new Error('Failed to get canvas context');
    }

    // Draw original image
    try {
      ctx.drawImage(image, 0, 0);
      console.log('[MangaTranslate Cache] Original image drawn on canvas');
    } catch (error) {
      console.error('[MangaTranslate Cache] Failed to draw image on canvas:', error);
      throw error;
    }

    // Render each translated bubble
    for (const region of regions) {
      this.renderBubble(ctx, region);
    }

    console.log('[MangaTranslate Cache] All bubbles rendered');

    // Convert to Blob
    const blob = await this.canvasToBlob(canvas);
    const translatedUrl = URL.createObjectURL(blob);

    console.log('[MangaTranslate Cache] Created blob URL:', translatedUrl);

    // Cache
    this.memoryCache.set(hash, {
      originalSrc: image.src,
      translatedBlob: blob,
      translatedUrl,
      regions,
      width: image.naturalWidth,
      height: image.naturalHeight,
      timestamp: Date.now(),
    });

    this.enforceMemoryLimit();
    return translatedUrl;
  }

  /**
   * Render single bubble onto canvas.
   */
  private renderBubble(ctx: CanvasRenderingContext2D, region: Region) {
    const [x, y, w, h] = region.bbox;
    const cx = x + w / 2;
    const cy = y + h / 2;

    console.log('[MangaTranslate Cache] Rendering bubble:', region.id, 'at', x, y, w, h);
    console.log('[MangaTranslate Cache] Text:', region.src_text, '->', region.tgt_text);

    // Step 1: Draw white mask
    ctx.save();
    ctx.fillStyle = '#FFFFFF';

    if (region.polygon && region.polygon.length >= 3) {
      ctx.beginPath();
      ctx.moveTo(region.polygon[0][0], region.polygon[0][1]);
      for (let i = 1; i < region.polygon.length; i++) {
        ctx.lineTo(region.polygon[i][0], region.polygon[i][1]);
      }
      ctx.closePath();
      ctx.fill();
    } else {
      switch (region.mask_type) {
        case 'rect':
          ctx.fillRect(x + 2, y + 2, w - 4, h - 4);
          break;
        case 'ellipse':
        default:
          ctx.beginPath();
          ctx.ellipse(cx, cy, (w / 2) * 0.92, (h / 2) * 0.92, 0, 0, Math.PI * 2);
          ctx.fill();
          break;
      }
    }
    ctx.restore();

    // Step 2: Draw translated text
    const text = region.tgt_text;
    if (!text || text.trim().length === 0) {
      console.log('[MangaTranslate Cache] No text to render for region:', region.id);
      return;
    }

    const fontSize = region.render?.font_size || 14;

    ctx.save();
    ctx.fillStyle = '#000000';
    ctx.font = `500 ${fontSize}px "Noto Sans", "Segoe UI", Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Word wrap
    const maxWidth = w * 0.85;
    const lines = this.wrapText(ctx, text, maxWidth);
    const lineHeight = fontSize * (region.render?.line_height || 1.25);

    // Center vertically
    const totalHeight = lines.length * lineHeight;
    const startY = cy - totalHeight / 2 + lineHeight / 2;

    // Draw each line
    for (let i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], cx, startY + i * lineHeight);
    }

    ctx.restore();
    console.log('[MangaTranslate Cache] Rendered', lines.length, 'lines of text');
  }

  /**
   * Word wrap for canvas text.
   */
  private wrapText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
    const isCJK = /[\u3000-\u9fff\uac00-\ud7af]/.test(text);

    if (isCJK) {
      return this.wrapCJKText(ctx, text, maxWidth);
    } else {
      return this.wrapLatinText(ctx, text, maxWidth);
    }
  }

  private wrapCJKText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
    const lines: string[] = [];
    let currentLine = '';

    for (const char of text) {
      const testLine = currentLine + char;
      const metrics = ctx.measureText(testLine);

      if (metrics.width > maxWidth && currentLine.length > 0) {
        lines.push(currentLine);
        currentLine = char;
      } else {
        currentLine = testLine;
      }
    }

    if (currentLine) {
      lines.push(currentLine);
    }

    return lines;
  }

  private wrapLatinText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
    const words = text.split(' ');
    const lines: string[] = [];
    let currentLine = '';

    for (const word of words) {
      const testLine = currentLine ? `${currentLine} ${word}` : word;
      const metrics = ctx.measureText(testLine);

      if (metrics.width > maxWidth && currentLine.length > 0) {
        lines.push(currentLine);
        currentLine = word;
      } else {
        currentLine = testLine;
      }
    }

    if (currentLine) {
      lines.push(currentLine);
    }

    return lines;
  }

  private canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => {
          if (blob) {
            console.log('[MangaTranslate Cache] Canvas converted to blob, size:', blob.size);
            resolve(blob);
          } else {
            reject(new Error('Failed to create blob from canvas'));
          }
        },
        'image/png',
        1.0
      );
    });
  }

  get(hash: string): string | null {
    const cached = this.memoryCache.get(hash);
    if (cached) {
      cached.timestamp = Date.now();
      return cached.translatedUrl;
    }
    return null;
  }

  getOriginal(hash: string): string | null {
    return this.memoryCache.get(hash)?.originalSrc || null;
  }

  has(hash: string): boolean {
    return this.memoryCache.has(hash);
  }

  private enforceMemoryLimit() {
    if (this.memoryCache.size <= this.maxMemoryItems) return;

    const entries = Array.from(this.memoryCache.entries()).sort(
      (a, b) => a[1].timestamp - b[1].timestamp
    );

    const toRemove = entries.slice(0, entries.length - this.maxMemoryItems);

    for (const [hash, cached] of toRemove) {
      URL.revokeObjectURL(cached.translatedUrl);
      this.memoryCache.delete(hash);
    }
  }

  clear() {
    for (const cached of this.memoryCache.values()) {
      URL.revokeObjectURL(cached.translatedUrl);
    }
    this.memoryCache.clear();
  }

  getStats() {
    return {
      count: this.memoryCache.size,
      maxItems: this.maxMemoryItems,
    };
  }
}
