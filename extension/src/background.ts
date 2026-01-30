/**
 * Background Service Worker - Unified API
 * Single endpoint with dynamic feature toggles.
 * Supports batch processing for parallel image translation.
 */

interface Settings {
  backendUrl: string;
  sourceLang: string;
  targetLang: string;
  // Feature toggles
  useSam: boolean;
  useAdvanced: boolean;
  detectOsb: boolean;
  useInpainting: boolean;
  inpaintMethod: 'opencv' | 'canvas';
  renderText: boolean;
  // Batch settings
  batchSize: number;
}

const DEFAULT_SETTINGS: Settings = {
  backendUrl: 'http://127.0.0.1:5000',
  sourceLang: 'ja',
  targetLang: 'id',
  // Feature toggles - sensible defaults
  useSam: false,
  useAdvanced: false,
  detectOsb: true,
  useInpainting: true,
  inpaintMethod: 'opencv',
  renderText: true,
  // Batch settings
  batchSize: 4,
};

interface BatchImageData {
  image_id: string;
  image_b64: string;
}

interface BatchRequest {
  images: BatchImageData[];
  source_lang: string;
  target_lang: string;
  detect_osb: boolean;
  use_inpainting: boolean;
  inpaint_method: string;
  use_sam: boolean;
  use_advanced: boolean;
  render_text: boolean;
}

interface TranslateRequest {
  action: 'translate' | 'translateBatch' | 'fetchImageAsBase64' | 'getSettings' | 'saveSettings' | 'checkHealth';
  data?: unknown;
}

// Message handler
chrome.runtime.onMessage.addListener((message: TranslateRequest, sender, sendResponse) => {
  handleMessage(message, sender)
    .then(sendResponse)
    .catch((error) => sendResponse({ success: false, error: error.message }));
  return true; // Keep channel open for async response
});

async function handleMessage(message: TranslateRequest, _sender: chrome.runtime.MessageSender) {
  const settings = await getSettings();

  switch (message.action) {
    case 'checkHealth':
      return checkHealth(settings);

    case 'translate':
      return translateImage(settings, message.data as {
        imageHash: string;
        imageBase64: string;
        sourceLang?: string;
        targetLang?: string;
        useSam?: boolean;
        useAdvanced?: boolean;
        detectOsb?: boolean;
        useInpainting?: boolean;
        inpaintMethod?: string;
        renderText?: boolean;
      });

    case 'translateBatch':
      return translateBatch(settings, message.data as BatchRequest);

    case 'fetchImageAsBase64':
      return fetchImageAsBase64((message.data as { url: string }).url);

    case 'getSettings':
      return { success: true, settings };

    case 'saveSettings':
      await chrome.storage.local.set({ settings: message.data });
      return { success: true };

    default:
      return { success: false, error: 'Unknown action' };
  }
}

async function getSettings(): Promise<Settings> {
  const result = await chrome.storage.local.get('settings');
  return { ...DEFAULT_SETTINGS, ...result.settings };
}

async function checkHealth(settings: Settings) {
  try {
    const response = await fetch(`${settings.backendUrl}/api/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    return { success: true, health: data };
  } catch (error: unknown) {
    const err = error as Error;
    return { success: false, error: err.message };
  }
}

/**
 * Unified translation - single endpoint with all features
 */
async function translateImage(settings: Settings, data: {
  imageHash: string;
  imageBase64: string;
  sourceLang?: string;
  targetLang?: string;
  useSam?: boolean;
  useAdvanced?: boolean;
  detectOsb?: boolean;
  useInpainting?: boolean;
  inpaintMethod?: string;
  renderText?: boolean;
}) {
  try {
    const response = await fetch(`${settings.backendUrl}/api/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_b64: data.imageBase64,
        source_lang: data.sourceLang || settings.sourceLang,
        target_lang: data.targetLang || settings.targetLang,
        // Feature toggles - use provided or settings
        use_sam: data.useSam ?? settings.useSam,
        use_advanced: data.useAdvanced ?? settings.useAdvanced,
        detect_osb: data.detectOsb ?? settings.detectOsb,
        use_inpainting: data.useInpainting ?? settings.useInpainting,
        inpaint_method: data.inpaintMethod || settings.inpaintMethod,
        render_text: data.renderText ?? settings.renderText,
        return_image: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();

    if (!result.success) {
      return { success: false, error: result.error || 'Translation failed' };
    }

    return {
      success: true,
      regions: result.regions || [],
      image_b64: result.image_b64,
      meta: result.meta || {},
    };
  } catch (error: unknown) {
    const err = error as Error;
    return { success: false, error: err.message };
  }
}

/**
 * Batch translation - process multiple images in parallel
 */
async function translateBatch(settings: Settings, data: BatchRequest) {
  try {
    const response = await fetch(`${settings.backendUrl}/api/batch_translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        images: data.images,
        source_lang: data.source_lang || settings.sourceLang,
        target_lang: data.target_lang || settings.targetLang,
        detect_osb: data.detect_osb ?? settings.detectOsb,
        use_inpainting: data.use_inpainting ?? settings.useInpainting,
        inpaint_method: data.inpaint_method || settings.inpaintMethod,
        use_sam: data.use_sam ?? settings.useSam,
        use_advanced: data.use_advanced ?? settings.useAdvanced,
        render_text: data.render_text ?? settings.renderText,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();

    if (!result.success) {
      return { success: false, error: result.error || 'Batch translation failed' };
    }

    return {
      success: true,
      results: result.results || [],
      meta: result.meta || {},
    };
  } catch (error: unknown) {
    const err = error as Error;
    return { success: false, error: err.message };
  }
}

async function fetchImageAsBase64(url: string) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const blob = await response.blob();

    return new Promise<{ success: boolean; base64?: string; error?: string }>((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        resolve({ success: true, base64: reader.result as string });
      };
      reader.onerror = () => {
        resolve({ success: false, error: 'Failed to read blob' });
      };
      reader.readAsDataURL(blob);
    });
  } catch (error: unknown) {
    const err = error as Error;
    return { success: false, error: err.message };
  }
}

console.log('[MangaTranslate] Background service worker started (Unified API + Batch)');
