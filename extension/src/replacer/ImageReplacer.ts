/**
 * Manages image src replacement with translated versions.
 */

import { TranslatedImageCache, Region } from '../cache/TranslatedImageCache';

interface ReplacedImageInfo {
  hash: string;
  originalSrc: string;
  isShowingTranslated: boolean;
}

export class ImageReplacer {
  private cache: TranslatedImageCache;
  private replacedImages: Map<HTMLImageElement, ReplacedImageInfo> = new Map();

  constructor() {
    this.cache = new TranslatedImageCache();
  }

  /**
   * Replace image src with translated version.
   * Uses a clean (CORS-free) image for canvas rendering.
   *
   * @param targetImage - The original img element on the page (will have src replaced)
   * @param cleanImage - A clean image loaded from base64 (for canvas rendering)
   * @param regions - Translation regions from backend
   * @param hash - Unique hash for caching
   */
  async replaceWithBase64(
    targetImage: HTMLImageElement,
    cleanImage: HTMLImageElement,
    regions: Region[],
    hash: string
  ): Promise<boolean> {
    try {
      // Skip if already replaced and showing translated
      const existing = this.replacedImages.get(targetImage);
      if (existing?.isShowingTranslated) {
        return true;
      }

      // Store original src
      const originalSrc = existing?.originalSrc || targetImage.src;

      // Render and cache if not already cached
      let translatedUrl = this.cache.get(hash);
      if (!translatedUrl) {
        // Use the clean image for canvas rendering (no CORS issues)
        translatedUrl = await this.cache.renderAndCache(cleanImage, regions, hash);
      }

      if (!translatedUrl) {
        console.error('[MangaTranslate] Failed to render translated image');
        return false;
      }

      // Replace src on the target image
      targetImage.src = translatedUrl;

      // Track replacement
      this.replacedImages.set(targetImage, {
        hash,
        originalSrc,
        isShowingTranslated: true,
      });

      // Add CSS class
      targetImage.classList.remove('manga-processing', 'manga-original');
      targetImage.classList.add('manga-translated');
      targetImage.dataset.translationHash = hash;

      console.log('[MangaTranslate] Image replaced successfully:', hash);
      return true;
    } catch (error) {
      console.error('[MangaTranslate] Failed to replace image:', error);
      return false;
    }
  }

  /**
   * Legacy replace method - may have CORS issues.
   * Use replaceWithBase64 instead.
   */
  async replace(
    image: HTMLImageElement,
    regions: Region[],
    hash: string
  ): Promise<boolean> {
    return this.replaceWithBase64(image, image, regions, hash);
  }

  /**
   * Restore image to original.
   */
  restore(image: HTMLImageElement): boolean {
    const info = this.replacedImages.get(image);
    if (!info) return false;

    image.src = info.originalSrc;
    info.isShowingTranslated = false;

    image.classList.remove('manga-translated');
    image.classList.add('manga-original');

    return true;
  }

  /**
   * Toggle between translated and original.
   */
  toggle(image: HTMLImageElement): boolean {
    const info = this.replacedImages.get(image);
    if (!info) return false;

    if (info.isShowingTranslated) {
      image.src = info.originalSrc;
      info.isShowingTranslated = false;
      image.classList.remove('manga-translated');
      image.classList.add('manga-original');
    } else {
      const translatedUrl = this.cache.get(info.hash);
      if (translatedUrl) {
        image.src = translatedUrl;
        info.isShowingTranslated = true;
        image.classList.add('manga-translated');
        image.classList.remove('manga-original');
      }
    }

    return info.isShowingTranslated;
  }

  /**
   * Toggle all images.
   */
  toggleAll(): boolean {
    let anyShowingTranslated = false;

    for (const info of this.replacedImages.values()) {
      if (info.isShowingTranslated) {
        anyShowingTranslated = true;
        break;
      }
    }

    const targetState = !anyShowingTranslated;

    for (const [image, info] of this.replacedImages) {
      if (targetState && !info.isShowingTranslated) {
        const translatedUrl = this.cache.get(info.hash);
        if (translatedUrl) {
          image.src = translatedUrl;
          info.isShowingTranslated = true;
          image.classList.add('manga-translated');
          image.classList.remove('manga-original');
        }
      } else if (!targetState && info.isShowingTranslated) {
        image.src = info.originalSrc;
        info.isShowingTranslated = false;
        image.classList.remove('manga-translated');
        image.classList.add('manga-original');
      }
    }

    return targetState;
  }

  /**
   * Restore all and clear cache.
   */
  restoreAll() {
    for (const [image] of this.replacedImages) {
      this.restore(image);
    }
    this.replacedImages.clear();
    this.cache.clear();
  }

  isReplaced(image: HTMLImageElement): boolean {
    return this.replacedImages.has(image);
  }

  isShowingTranslated(image: HTMLImageElement): boolean {
    return this.replacedImages.get(image)?.isShowingTranslated || false;
  }

  getStats() {
    let translated = 0;
    let original = 0;

    for (const info of this.replacedImages.values()) {
      if (info.isShowingTranslated) translated++;
      else original++;
    }

    return {
      total: this.replacedImages.size,
      showingTranslated: translated,
      showingOriginal: original,
      cache: this.cache.getStats(),
    };
  }
}
