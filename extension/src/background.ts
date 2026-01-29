/**
 * Background Service Worker
 * Handles API calls and CORS bypass for the extension.
 */

interface TranslateRequest {
  action: 'translate' | 'translateImage' | 'batchDetect' | 'batchTranslate' |
          'fetchImageAsBase64' | 'getSettings' | 'saveSettings' | 'checkHealth' | 'updateProgress';
  data?: unknown;
}

interface Settings {
  backendUrl: string;
  sourceLang: string;
  targetLang: string;
}

const DEFAULT_SETTINGS: Settings = {
  backendUrl: 'http://127.0.0.1:5000',
  sourceLang: 'ja',
  targetLang: 'id',
};

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

    case 'translateImage':
      return translateImage(settings, message.data as {
        imageHash: string;
        imageBase64: string;
        sourceLang?: string;
        targetLang?: string;
      });

    case 'batchDetect':
      return batchDetect(settings, message.data as {
        images: Array<{ image_id: string; image_b64: string }>;
        sourceLang?: string;
      });

    case 'batchTranslate':
      return batchTranslate(settings, message.data as {
        bubbles: unknown[];
        sourceLang?: string;
        targetLang?: string;
      });

    case 'fetchImageAsBase64':
      return fetchImageAsBase64((message.data as { url: string }).url);

    case 'getSettings':
      return { success: true, settings };

    case 'saveSettings':
      await chrome.storage.local.set({ settings: message.data });
      return { success: true };

    case 'updateProgress':
      // Relay to popup if open
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
    const response = await fetch(`${settings.backendUrl}/health`, {
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

async function translateImage(settings: Settings, data: {
  imageHash: string;
  imageBase64: string;
  sourceLang?: string;
  targetLang?: string;
}) {
  try {
    const response = await fetch(`${settings.backendUrl}/v1/translate_viewport`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_b64: data.imageBase64,
        source_lang: data.sourceLang || settings.sourceLang,
        target_lang: data.targetLang || settings.targetLang,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();
    return {
      success: true,
      regions: result.regions || [],
      meta: result.meta || {},
    };
  } catch (error: unknown) {
    const err = error as Error;
    return { success: false, error: err.message };
  }
}

async function batchDetect(settings: Settings, data: {
  images: Array<{ image_id: string; image_b64: string }>;
  sourceLang?: string;
}) {
  try {
    const response = await fetch(`${settings.backendUrl}/v1/batch_detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        images: data.images,
        source_lang: data.sourceLang || settings.sourceLang,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();
    return {
      success: true,
      bubbles: result.bubbles || [],
      meta: {
        total_images: result.total_images,
        total_bubbles: result.total_bubbles,
        detect_ms: result.detect_ms,
        ocr_ms: result.ocr_ms,
      },
    };
  } catch (error: unknown) {
    const err = error as Error;
    return { success: false, error: err.message };
  }
}

async function batchTranslate(settings: Settings, data: {
  bubbles: unknown[];
  sourceLang?: string;
  targetLang?: string;
}) {
  try {
    const response = await fetch(`${settings.backendUrl}/v1/batch_translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        bubbles: data.bubbles,
        source_lang: data.sourceLang || settings.sourceLang,
        target_lang: data.targetLang || settings.targetLang,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();
    return {
      success: true,
      bubbles: result.bubbles || [],
      meta: {
        total_translated: result.total_translated,
        translate_ms: result.translate_ms,
        cached_hits: result.cached_hits,
      },
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
    const reader = new FileReader();

    return new Promise<{ success: boolean; base64?: string; error?: string }>((resolve) => {
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

// Log when service worker starts
console.log('[MangaTranslate] Background service worker started');
