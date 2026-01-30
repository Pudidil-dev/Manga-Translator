/**
 * Content Script - Unified API
 * Detects manga images and processes them with user-selected features.
 */

import { ImageReplacer } from './replacer/ImageReplacer';
import { Region } from './cache/TranslatedImageCache';

interface Settings {
  backendUrl: string;
  sourceLang: string;
  targetLang: string;
  useSam: boolean;
  useAdvanced: boolean;
  detectOsb: boolean;
  useInpainting: boolean;
  inpaintMethod: 'opencv' | 'canvas';
  renderText: boolean;
}

const imageReplacer = new ImageReplacer();

let isProcessing = false;
let currentSettings: Settings | null = null;
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
 * Convert image to base64 via background script (CORS bypass).
 */
async function imageToBase64(img: HTMLImageElement): Promise<string> {
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
    if (!base64.startsWith('data:')) {
      img.src = 'data:image/png;base64,' + base64;
    } else {
      img.src = base64;
    }
  });
}

/**
 * Get current settings from background.
 */
async function getSettings(): Promise<Settings> {
  const response = await chrome.runtime.sendMessage({ action: 'getSettings' });
  if (response?.success) {
    return response.settings;
  }
  // Defaults
  return {
    backendUrl: 'http://127.0.0.1:5000',
    sourceLang: 'ja',
    targetLang: 'id',
    useSam: false,
    useAdvanced: false,
    detectOsb: true,
    useInpainting: true,
    inpaintMethod: 'opencv',
    renderText: true,
  };
}

/**
 * Process single image using unified API.
 */
async function processImage(image: HTMLImageElement): Promise<boolean> {
  const hash = hashImage(image);

  try {
    const base64 = await imageToBase64(image);

    const response = await chrome.runtime.sendMessage({
      action: 'translate',
      data: {
        imageHash: hash,
        imageBase64: base64,
        // Use current settings (background will merge with defaults)
      },
    });

    if (!response?.success) {
      console.log('[MangaTranslate] Translation failed:', response?.error);
      return false;
    }

    if (!response.regions?.length) {
      console.log('[MangaTranslate] No text regions found');
      return false;
    }

    // If backend returned processed image, use it
    if (response.image_b64) {
      try {
        const processedImage = await loadImageFromBase64(response.image_b64);
        return await imageReplacer.replaceWithInpainted(
          image,
          processedImage,
          response.regions as Region[],
          hash
        );
      } catch (err) {
        console.log('[MangaTranslate] Failed to load processed image, using canvas');
      }
    }

    // Fallback: render on canvas
    const cleanImage = await loadImageFromBase64(base64);
    return await imageReplacer.replaceWithBase64(
      image,
      cleanImage,
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
  // Could send to popup if needed
  console.log(`[MangaTranslate] ${message} (${percent}%)`);
}

/**
 * Translate all manga images on page.
 */
async function translateAllImages() {
  if (isProcessing) return;
  isProcessing = true;

  updateProgress('Loading settings...', 0);
  currentSettings = await getSettings();

  // Log active features
  const features: string[] = [];
  if (currentSettings.useSam) features.push('SAM2');
  if (currentSettings.useAdvanced) features.push('Advanced');
  if (currentSettings.detectOsb) features.push('OSB');
  if (currentSettings.useInpainting) features.push(`Inpaint:${currentSettings.inpaintMethod}`);
  console.log(`[MangaTranslate] Features: ${features.join(', ') || 'default'}`);

  updateProgress('Finding images...', 5);
  const images = findMangaImages();

  if (images.length === 0) {
    updateProgress('No manga images found', 100);
    isProcessing = false;
    return;
  }

  totalImages = images.length;
  completedImages = 0;
  updateProgress(`Found ${totalImages} images...`, 10);

  // Process images (one at a time for stability)
  for (const img of images) {
    const success = await processImage(img);
    completedImages++;
    updateProgress(
      `Translating ${completedImages}/${totalImages}${success ? '' : ' (skipped)'}`,
      Math.round(10 + (completedImages / totalImages) * 90)
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
    outline: 2px solid #4CAF50;
    outline-offset: -2px;
  }
  .manga-original {
    outline: 2px dashed #888;
    outline-offset: -2px;
  }
  .manga-processing {
    outline: 2px solid #FF9800;
    outline-offset: -2px;
    animation: pulse 1s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
`;
document.head.appendChild(style);

console.log('[MangaTranslate] Content script loaded (Unified API)');
