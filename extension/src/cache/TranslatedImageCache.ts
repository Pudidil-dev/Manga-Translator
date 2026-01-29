/**
 * Cache for rendered translated images.
 * Uses Canvas API to render text onto images and stores as Blob.
 *
 * Features:
 * - Collision detection between text regions
 * - Auto font sizing to fit bubble
 * - Smart text wrapping for CJK and Latin
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

interface TextBounds {
  x: number;
  y: number;
  width: number;
  height: number;
  regionId: string;
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

  // Minimum gap between text regions (pixels)
  private readonly MIN_TEXT_GAP = 4;
  // Minimum font size
  private readonly MIN_FONT_SIZE = 8;
  // Maximum font size
  private readonly MAX_FONT_SIZE = 28;

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

    // Sort regions by Y position (top to bottom) for proper layering
    const sortedRegions = [...regions].sort((a, b) => a.bbox[1] - b.bbox[1]);

    // First pass: Draw all white masks
    for (const region of sortedRegions) {
      this.drawMask(ctx, region);
    }

    // Calculate optimal font sizes considering collisions
    const optimizedRegions = this.optimizeFontSizes(ctx, sortedRegions);

    // Second pass: Draw all text
    for (const region of optimizedRegions) {
      this.renderText(ctx, region);
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
   * Optimize font sizes to prevent text overlap between regions.
   */
  private optimizeFontSizes(ctx: CanvasRenderingContext2D, regions: Region[]): Region[] {
    const optimized: Region[] = [];
    const renderedBounds: TextBounds[] = [];

    for (const region of regions) {
      const [x, y, w, h] = region.bbox;
      const text = region.tgt_text;

      if (!text || text.trim().length === 0) {
        optimized.push(region);
        continue;
      }

      // Calculate available space considering mask type
      const availableWidth = this.getAvailableWidth(region);
      const availableHeight = this.getAvailableHeight(region);

      // Start with suggested font size or calculate based on area
      let fontSize = Math.min(
        region.render?.font_size || 14,
        this.MAX_FONT_SIZE,
        Math.floor(availableHeight / 3),
        Math.floor(availableWidth / 3)
      );

      // Try to fit text, reducing font size if needed
      let textBounds: TextBounds | null = null;
      let bestLines: string[] = [];

      while (fontSize >= this.MIN_FONT_SIZE) {
        ctx.font = `500 ${fontSize}px "Noto Sans", "Segoe UI", Arial, sans-serif`;
        const lineHeight = fontSize * (region.render?.line_height || 1.2);

        const lines = this.wrapText(ctx, text, availableWidth * 0.9);
        const totalTextHeight = lines.length * lineHeight;

        // Check if text fits in bubble
        if (totalTextHeight <= availableHeight * 0.9) {
          // Calculate text bounds
          const maxLineWidth = Math.max(...lines.map(line => ctx.measureText(line).width));
          const cx = x + w / 2;
          const cy = y + h / 2;

          textBounds = {
            x: cx - maxLineWidth / 2 - this.MIN_TEXT_GAP,
            y: cy - totalTextHeight / 2 - this.MIN_TEXT_GAP,
            width: maxLineWidth + this.MIN_TEXT_GAP * 2,
            height: totalTextHeight + this.MIN_TEXT_GAP * 2,
            regionId: region.id
          };

          // Check for collision with already rendered regions
          const hasCollision = renderedBounds.some(bound =>
            this.checkCollision(textBounds!, bound)
          );

          if (!hasCollision) {
            bestLines = lines;
            break;
          }
        }

        fontSize -= 1;
      }

      // If we found a valid size, record the bounds
      if (textBounds && fontSize >= this.MIN_FONT_SIZE) {
        renderedBounds.push(textBounds);
      }

      // Create optimized region with new font size
      optimized.push({
        ...region,
        render: {
          ...region.render,
          font_size: Math.max(fontSize, this.MIN_FONT_SIZE),
        }
      });
    }

    return optimized;
  }

  /**
   * Check if two text bounds collide.
   */
  private checkCollision(a: TextBounds, b: TextBounds): boolean {
    return !(
      a.x + a.width < b.x ||
      b.x + b.width < a.x ||
      a.y + a.height < b.y ||
      b.y + b.height < a.y
    );
  }

  /**
   * Get available width inside bubble (accounting for mask shape).
   */
  private getAvailableWidth(region: Region): number {
    const [, , w] = region.bbox;

    switch (region.mask_type) {
      case 'ellipse':
        return w * 0.7; // Ellipse has less usable width
      case 'rect':
        return w * 0.85;
      default:
        return w * 0.75;
    }
  }

  /**
   * Get available height inside bubble (accounting for mask shape).
   */
  private getAvailableHeight(region: Region): number {
    const [, , , h] = region.bbox;

    switch (region.mask_type) {
      case 'ellipse':
        return h * 0.65; // Ellipse has less usable height
      case 'rect':
        return h * 0.85;
      default:
        return h * 0.7;
    }
  }

  /**
   * Draw white mask for bubble.
   */
  private drawMask(ctx: CanvasRenderingContext2D, region: Region) {
    const [x, y, w, h] = region.bbox;
    const cx = x + w / 2;
    const cy = y + h / 2;

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
          // Add small padding to avoid edge artifacts
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
  }

  /**
   * Render text onto canvas.
   */
  private renderText(ctx: CanvasRenderingContext2D, region: Region) {
    const [x, y, w, h] = region.bbox;
    const cx = x + w / 2;
    const cy = y + h / 2;
    const text = region.tgt_text;

    if (!text || text.trim().length === 0) {
      return;
    }

    // Use optimized font size
    const fontSize = Math.max(region.render?.font_size || 14, this.MIN_FONT_SIZE);
    const lineHeight = fontSize * (region.render?.line_height || 1.2);

    ctx.save();
    ctx.fillStyle = '#000000';
    ctx.font = `500 ${fontSize}px "Noto Sans", "Segoe UI", Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Get available width for text
    const maxWidth = this.getAvailableWidth(region);
    const lines = this.wrapText(ctx, text, maxWidth);

    // Calculate total height and starting position
    const totalHeight = lines.length * lineHeight;
    const availableHeight = this.getAvailableHeight(region);

    // Ensure text stays within bubble
    const clampedHeight = Math.min(totalHeight, availableHeight);
    const startY = cy - clampedHeight / 2 + lineHeight / 2;

    // Draw each line
    for (let i = 0; i < lines.length; i++) {
      const lineY = startY + i * lineHeight;

      // Skip lines that would go outside the bubble
      if (lineY < y + h * 0.1 || lineY > y + h * 0.9) {
        continue;
      }

      ctx.fillText(lines[i], cx, lineY);
    }

    ctx.restore();
  }

  /**
   * Word wrap for canvas text.
   */
  private wrapText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
    // Handle empty or very short text
    if (!text || text.trim().length === 0) {
      return [];
    }

    // Check if text contains CJK characters
    const isCJK = /[\u3000-\u9fff\uac00-\ud7af]/.test(text);

    if (isCJK) {
      return this.wrapCJKText(ctx, text, maxWidth);
    } else {
      return this.wrapLatinText(ctx, text, maxWidth);
    }
  }

  /**
   * Wrap CJK text (character by character).
   */
  private wrapCJKText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
    const lines: string[] = [];
    let currentLine = '';

    for (const char of text) {
      // Handle newlines
      if (char === '\n') {
        if (currentLine) {
          lines.push(currentLine);
        }
        currentLine = '';
        continue;
      }

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

  /**
   * Wrap Latin text (word by word).
   */
  private wrapLatinText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
    // Split on whitespace but preserve structure
    const words = text.replace(/\n/g, ' ').split(/\s+/).filter(w => w.length > 0);
    const lines: string[] = [];
    let currentLine = '';

    for (const word of words) {
      const testLine = currentLine ? `${currentLine} ${word}` : word;
      const metrics = ctx.measureText(testLine);

      if (metrics.width > maxWidth && currentLine.length > 0) {
        lines.push(currentLine);

        // Check if single word is too long, break it
        if (ctx.measureText(word).width > maxWidth) {
          const brokenLines = this.breakLongWord(ctx, word, maxWidth);
          lines.push(...brokenLines.slice(0, -1));
          currentLine = brokenLines[brokenLines.length - 1] || '';
        } else {
          currentLine = word;
        }
      } else {
        currentLine = testLine;
      }
    }

    if (currentLine) {
      lines.push(currentLine);
    }

    return lines;
  }

  /**
   * Break a long word that doesn't fit in maxWidth.
   */
  private breakLongWord(ctx: CanvasRenderingContext2D, word: string, maxWidth: number): string[] {
    const lines: string[] = [];
    let currentPart = '';

    for (const char of word) {
      const testPart = currentPart + char;
      if (ctx.measureText(testPart).width > maxWidth && currentPart.length > 0) {
        lines.push(currentPart);
        currentPart = char;
      } else {
        currentPart = testPart;
      }
    }

    if (currentPart) {
      lines.push(currentPart);
    }

    return lines;
  }

  private canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => {
          if (blob) {
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
