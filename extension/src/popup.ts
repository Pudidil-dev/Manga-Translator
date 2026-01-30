/**
 * Popup UI Script - Unified API
 * Interactive feature checkboxes, no modes.
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
}

// DOM Elements
const sourceLangEl = document.getElementById('sourceLang') as HTMLSelectElement;
const targetLangEl = document.getElementById('targetLang') as HTMLSelectElement;
const backendUrlEl = document.getElementById('backendUrl') as HTMLInputElement;

// Feature toggles
const detectOsbEl = document.getElementById('detectOsb') as HTMLInputElement;
const useInpaintingEl = document.getElementById('useInpainting') as HTMLInputElement;
const inpaintMethodEl = document.getElementById('inpaintMethod') as HTMLSelectElement;
const inpaintMethodGroup = document.getElementById('inpaintMethodGroup') as HTMLDivElement;
const useSamEl = document.getElementById('useSam') as HTMLInputElement;
const useAdvancedEl = document.getElementById('useAdvanced') as HTMLInputElement;
const renderTextEl = document.getElementById('renderText') as HTMLInputElement;

// Actions
const translateBtn = document.getElementById('translateBtn') as HTMLButtonElement;
const toggleBtn = document.getElementById('toggleBtn') as HTMLButtonElement;
const clearBtn = document.getElementById('clearBtn') as HTMLButtonElement;

// Status
const statusText = document.getElementById('statusText') as HTMLDivElement;
const progressFill = document.getElementById('progressFill') as HTMLDivElement;
const statsEl = document.getElementById('stats') as HTMLDivElement;

// Collapsible
const advancedHeader = document.getElementById('advancedHeader') as HTMLHeadingElement;
const advancedContent = document.getElementById('advancedContent') as HTMLDivElement;

// Load settings
async function loadSettings() {
  const response = await chrome.runtime.sendMessage({ action: 'getSettings' });
  if (response?.success) {
    const settings: Settings = response.settings;
    sourceLangEl.value = settings.sourceLang || 'ja';
    targetLangEl.value = settings.targetLang || 'id';
    backendUrlEl.value = settings.backendUrl || 'http://127.0.0.1:5000';

    // Feature toggles
    detectOsbEl.checked = settings.detectOsb ?? true;
    useInpaintingEl.checked = settings.useInpainting ?? true;
    inpaintMethodEl.value = settings.inpaintMethod || 'opencv';
    useSamEl.checked = settings.useSam ?? false;
    useAdvancedEl.checked = settings.useAdvanced ?? false;
    renderTextEl.checked = settings.renderText ?? true;

    updateInpaintMethodVisibility();
  }
}

// Save settings
async function saveSettings() {
  const settings: Settings = {
    sourceLang: sourceLangEl.value,
    targetLang: targetLangEl.value,
    backendUrl: backendUrlEl.value,
    // Feature toggles
    detectOsb: detectOsbEl.checked,
    useInpainting: useInpaintingEl.checked,
    inpaintMethod: inpaintMethodEl.value as Settings['inpaintMethod'],
    useSam: useSamEl.checked,
    useAdvanced: useAdvancedEl.checked,
    renderText: renderTextEl.checked,
  };
  await chrome.runtime.sendMessage({ action: 'saveSettings', data: settings });
}

// Update inpaint method visibility
function updateInpaintMethodVisibility() {
  if (useInpaintingEl.checked) {
    inpaintMethodGroup.style.display = 'flex';
  } else {
    inpaintMethodGroup.style.display = 'none';
  }
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
      setStatus('Backend ready', 100);
      translateBtn.disabled = false;
    } else {
      setStatus('Backend unhealthy', 0);
      translateBtn.disabled = true;
    }
  } else {
    setStatus(`Backend offline`, 0);
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
    await chrome.tabs.sendMessage(tab.id, { action: 'translatePage' });
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
    setStatus('Cleared', 0);
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
      statsEl.textContent = `${stats.showingTranslated}/${stats.total} translated`;

      if (!isProcessing) {
        translateBtn.disabled = false;
      }
    }
  } catch {
    statsEl.textContent = '';
  }
}

// Toggle collapsible
function toggleAdvanced() {
  if (advancedContent.style.display === 'none') {
    advancedContent.style.display = 'block';
  } else {
    advancedContent.style.display = 'none';
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
advancedHeader.addEventListener('click', toggleAdvanced);

// Save settings on change
sourceLangEl.addEventListener('change', saveSettings);
targetLangEl.addEventListener('change', saveSettings);
backendUrlEl.addEventListener('blur', saveSettings);

// Feature toggles
detectOsbEl.addEventListener('change', saveSettings);
useInpaintingEl.addEventListener('change', () => {
  updateInpaintMethodVisibility();
  saveSettings();
});
inpaintMethodEl.addEventListener('change', saveSettings);
useSamEl.addEventListener('change', saveSettings);
useAdvancedEl.addEventListener('change', saveSettings);
renderTextEl.addEventListener('change', saveSettings);

// Initialize
advancedContent.style.display = 'none';
loadSettings();
checkHealth();
updateStats();

// Refresh stats periodically
setInterval(updateStats, 2000);
