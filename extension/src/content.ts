/**
 * Content Script
 * Detects manga images, processes them, and renders translations.
 */

import { ImageReplacer } from './replacer/ImageReplacer';
import { Region } from './cache/TranslatedImageCache';

const imageReplacer = new ImageReplacer();

let isProcessing = false;
let currentSourceLang = 'ja';
let currentTargetLang = 'id';
let totalImages = 0;
let completedImages = 0;

/**
 * Generate hash for image.
 */
function hashImage(image: HTMLImageElement): string {
  const key = `${image.src}_${image.naturalWidth}_${image.naturalHeight}`;
  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    hash = ((hash << 5) - hash) + key.charCodeAt(i);
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16);
}

/**
 * Find manga images on page.
 */
function findMangaImages(): HTMLImageElement[] {
  const allImages = document.querySelectorAll('img');
  const mangaImages: HTMLImageElement[] = [];

  allImages.forEach((img) => {
    // Must be loaded
    if (!img.complete || img.naturalWidth === 0) return;

    // Size check - manga pages are large
    if (img.naturalWidth < 300 || img.naturalHeight < 400) return;

    // Display size check
    if (img.width < 100 || img.height < 100) return;

    // Skip tiny data URIs
    if (img.src.startsWith('data:') && img.src.length < 1000) return;

    // Aspect ratio check
    const aspect = img.naturalWidth / img.naturalHeight;
    if (aspect < 0.3 || aspect > 1.5) return;

    // Skip already processed
    if (imageReplacer.isReplaced(img)) return;

    mangaImages.push(img);
  });

  return mangaImages;
}

/**
 * Convert image to base64.
 * Returns base64 string (always fetches via background to avoid CORS).
 */
async function imageToBase64(img: HTMLImageElement): Promise<string> {
  // Always fetch via background to avoid CORS issues with canvas
  const response = await chrome.runtime.sendMessage({
    action: 'fetchImageAsBase64',
    data: { url: img.src },
  });

  if (response?.success && response.base64) {
    return response.base64;
  }

  // Fallback: try canvas (same-origin only)
  try {
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0);
    return canvas.toDataURL('image/png');
  } catch {
    throw new Error('Could not convert image (CORS blocked)');
  }
}

/**
 * Load image from base64 string.
 */
function loadImageFromBase64(base64: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load image from base64'));
    img.src = base64;
  });
}

/**
 * Process single image.
 */
async function processImage(image: HTMLImageElement): Promise<boolean> {
  const hash = hashImage(image);

  try {
    // Get base64 (via background to bypass CORS)
    const base64 = await imageToBase64(image);

    // Send to backend for translation
    const response = await chrome.runtime.sendMessage({
      action: 'translateImage',
      data: {
        imageHash: hash,
        imageBase64: base64,
        sourceLang: currentSourceLang,
        targetLang: currentTargetLang,
      },
    });

    if (!response?.success) {
      console.log('[MangaTranslate] Translation failed:', response?.error);
      return false;
    }

    if (!response.regions?.length) {
      console.log('[MangaTranslate] No text regions found in image');
      return false;
    }

    // Create clean image from base64 (no CORS issues)
    const cleanImage = await loadImageFromBase64(base64);

    // Replace image with translated version
    // Pass the clean image for canvas rendering, and original image for src replacement
    return await imageReplacer.replaceWithBase64(
      image,           // Original img element (will have its src replaced)
      cleanImage,      // Clean image for canvas rendering
      response.regions as Region[],
      hash
    );
  } catch (error) {
    console.error('[MangaTranslate] Error processing image:', error);
    return false;
  }
}

/**
 * Update progress notification.
 */
function updateProgress(message: string, percent: number) {
  chrome.runtime.sendMessage({
    action: 'updateProgress',
    data: { message, percent },
  });
}

/**
 * Translate all manga images on page.
 */
async function translateAllImages() {
  if (isProcessing) return;
  isProcessing = true;

  updateProgress('Finding images...', 0);

  const images = findMangaImages();
  if (images.length === 0) {
    updateProgress('No manga images found', 100);
    isProcessing = false;
    return;
  }

  totalImages = images.length;
  completedImages = 0;

  updateProgress(`Found ${totalImages} images, starting translation...`, 5);

  // Process in batches
  const BATCH_SIZE = 2; // Reduced for stability

  for (let i = 0; i < images.length; i += BATCH_SIZE) {
    const batch = images.slice(i, i + BATCH_SIZE);

    await Promise.all(
      batch.map(async (img) => {
        const success = await processImage(img);
        completedImages++;
        updateProgress(
          `Translating ${completedImages}/${totalImages}${success ? '' : ' (skipped)'}`,
          Math.round((completedImages / totalImages) * 100)
        );
      })
    );
  }

  const stats = imageReplacer.getStats();
  updateProgress(`Done! ${stats.showingTranslated} images translated`, 100);
  isProcessing = false;
}

/**
 * Message handler.
 */
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  switch (message.action) {
    case 'translatePage':
      currentSourceLang = message.sourceLang || 'ja';
      currentTargetLang = message.targetLang || 'id';
      translateAllImages();
      sendResponse({ success: true });
      break;

    case 'toggleOverlay':
      const isTranslated = imageReplacer.toggleAll();
      sendResponse({ success: true, showingTranslated: isTranslated });
      break;

    case 'clearOverlays':
      imageReplacer.restoreAll();
      sendResponse({ success: true });
      break;

    case 'getStats':
      sendResponse({
        success: true,
        stats: imageReplacer.getStats(),
        isProcessing,
      });
      break;
  }
  return true;
});

// Inject styles
const style = document.createElement('style');
style.textContent = `
  .manga-translated {
    outline: 2px solid #e94560;
    outline-offset: -2px;
  }
  .manga-original {
    outline: 2px dashed #888;
    outline-offset: -2px;
  }
  .manga-processing {
    outline: 2px solid #ffa500;
    outline-offset: -2px;
    animation: pulse 1s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
`;
document.head.appendChild(style);

console.log('[MangaTranslate] Content script loaded');
