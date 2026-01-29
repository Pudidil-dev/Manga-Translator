/**
 * Popup UI Script
 */

interface Settings {
  backendUrl: string;
  sourceLang: string;
  targetLang: string;
}

// DOM Elements
const sourceLangEl = document.getElementById('sourceLang') as HTMLSelectElement;
const targetLangEl = document.getElementById('targetLang') as HTMLSelectElement;
const backendUrlEl = document.getElementById('backendUrl') as HTMLInputElement;
const translateBtn = document.getElementById('translateBtn') as HTMLButtonElement;
const toggleBtn = document.getElementById('toggleBtn') as HTMLButtonElement;
const clearBtn = document.getElementById('clearBtn') as HTMLButtonElement;
const statusText = document.getElementById('statusText') as HTMLDivElement;
const progressFill = document.getElementById('progressFill') as HTMLDivElement;
const statsEl = document.getElementById('stats') as HTMLDivElement;

// Load settings
async function loadSettings() {
  const response = await chrome.runtime.sendMessage({ action: 'getSettings' });
  if (response?.success) {
    const settings: Settings = response.settings;
    sourceLangEl.value = settings.sourceLang;
    targetLangEl.value = settings.targetLang;
    backendUrlEl.value = settings.backendUrl;
  }
}

// Save settings
async function saveSettings() {
  const settings: Settings = {
    sourceLang: sourceLangEl.value,
    targetLang: targetLangEl.value,
    backendUrl: backendUrlEl.value,
  };
  await chrome.runtime.sendMessage({ action: 'saveSettings', data: settings });
}

// Update status
function setStatus(message: string, percent: number = 0) {
  statusText.textContent = message;
  progressFill.style.width = `${percent}%`;
}

// Check backend health
async function checkHealth() {
  setStatus('Checking backend...', 0);
  const response = await chrome.runtime.sendMessage({ action: 'checkHealth' });

  if (response?.success) {
    const health = response.health;
    if (health.ok) {
      setStatus(`Backend ready (${health.yolo_format})`, 100);
      translateBtn.disabled = false;
    } else {
      setStatus('Backend unhealthy', 0);
      translateBtn.disabled = true;
    }
  } else {
    setStatus(`Backend offline: ${response?.error || 'Unknown'}`, 0);
    translateBtn.disabled = true;
  }
}

// Get current tab
async function getCurrentTab(): Promise<chrome.tabs.Tab | null> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab || null;
}

// Translate page
async function translatePage() {
  await saveSettings();

  const tab = await getCurrentTab();
  if (!tab?.id) {
    setStatus('No active tab', 0);
    return;
  }

  translateBtn.disabled = true;
  setStatus('Starting translation...', 10);

  try {
    await chrome.tabs.sendMessage(tab.id, {
      action: 'translatePage',
      sourceLang: sourceLangEl.value,
      targetLang: targetLangEl.value,
    });
    setStatus('Translation started', 20);
  } catch (error: unknown) {
    const err = error as Error;
    setStatus(`Error: ${err.message}`, 0);
    translateBtn.disabled = false;
  }
}

// Toggle overlay
async function toggleOverlay() {
  const tab = await getCurrentTab();
  if (!tab?.id) return;

  try {
    const response = await chrome.tabs.sendMessage(tab.id, { action: 'toggleOverlay' });
    if (response?.success) {
      setStatus(response.showingTranslated ? 'Showing translated' : 'Showing original', 100);
    }
  } catch (error: unknown) {
    const err = error as Error;
    setStatus(`Error: ${err.message}`, 0);
  }
}

// Clear overlays
async function clearOverlays() {
  const tab = await getCurrentTab();
  if (!tab?.id) return;

  try {
    await chrome.tabs.sendMessage(tab.id, { action: 'clearOverlays' });
    setStatus('Cleared all translations', 0);
  } catch (error: unknown) {
    const err = error as Error;
    setStatus(`Error: ${err.message}`, 0);
  }
}

// Update stats
async function updateStats() {
  const tab = await getCurrentTab();
  if (!tab?.id) return;

  try {
    const response = await chrome.tabs.sendMessage(tab.id, { action: 'getStats' });
    if (response?.success) {
      const { stats, isProcessing } = response;
      statsEl.textContent = `Images: ${stats.total} | Translated: ${stats.showingTranslated} | Cache: ${stats.cache.count}`;

      if (!isProcessing) {
        translateBtn.disabled = false;
      }
    }
  } catch {
    statsEl.textContent = 'No page data';
  }
}

// Listen for progress updates
chrome.runtime.onMessage.addListener((message) => {
  if (message.action === 'updateProgress') {
    setStatus(message.data.message, message.data.percent);
    if (message.data.percent >= 100) {
      translateBtn.disabled = false;
      updateStats();
    }
  }
});

// Event listeners
translateBtn.addEventListener('click', translatePage);
toggleBtn.addEventListener('click', toggleOverlay);
clearBtn.addEventListener('click', clearOverlays);

// Save settings on change
sourceLangEl.addEventListener('change', saveSettings);
targetLangEl.addEventListener('change', saveSettings);
backendUrlEl.addEventListener('blur', saveSettings);

// Initialize
loadSettings();
checkHealth();
updateStats();

// Refresh stats periodically
setInterval(updateStats, 2000);
