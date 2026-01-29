import logging
import os
from typing import Optional

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.0-flash"

LANGUAGE_NAMES = {
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "pt": "Portuguese",
    "ms": "Malay",
    "hi": "Hindi",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
}


class MangaTranslator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_url: Optional[str] = None,
        source: str = "ja",
        target: str = "en",
        timeout_seconds: float = 30.0,
    ):
        self.source = source
        self.target = target
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.base_url = api_url or os.getenv("GEMINI_API_URL", DEFAULT_API_URL)
        self.timeout = timeout_seconds
        self._cache: dict[str, str] = {}

    def translate(self, text: str, source_lang: Optional[str] = None, target_lang: Optional[str] = None) -> str:
        """
        Translate text using Gemini API.

        Args:
            text: Text to translate.
            source_lang: Source language code (default: self.source).
            target_lang: Target language code (default: self.target).
        """
        if not text:
            return text

        if not self.api_key and self.base_url == DEFAULT_API_URL:
            raise ValueError("GEMINI_API_KEY not set.")

        src = source_lang or self.source
        tgt = target_lang or self.target
        clean_text = self._preprocess_text(text)
        cache_key = self._make_cache_key(clean_text, src, tgt)
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = self._build_prompt(clean_text, src, tgt)

        try:
            translated = self._call_gemini(prompt)
            self._cache[cache_key] = translated
            return translated
        except Exception as exc:
            logger.warning("Gemini translation failed: %s", exc)
            return text

    def translate_batch(
        self,
        texts: list[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> list[str]:
        if not texts:
            return []

        if not self.api_key and self.base_url == DEFAULT_API_URL:
            raise ValueError("GEMINI_API_KEY not set.")

        src = source_lang or self.source
        tgt = target_lang or self.target

        results: list[Optional[str]] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_idx: list[int] = []

        for i, text in enumerate(texts):
            clean_text = self._preprocess_text(text)
            cache_key = self._make_cache_key(clean_text, src, tgt)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                missing_texts.append(clean_text)
                missing_idx.append(i)

        if not missing_texts:
            return [r or "" for r in results]

        prompt = self._build_batch_prompt(missing_texts, src, tgt)

        try:
            result = self._call_gemini(prompt)
            parsed = self._parse_batch_response(result, len(missing_texts))
            if parsed:
                for i, translated in enumerate(parsed):
                    idx = missing_idx[i]
                    results[idx] = translated
                    cache_key = self._make_cache_key(missing_texts[i], src, tgt)
                    self._cache[cache_key] = translated
                return [r or "" for r in results]
        except Exception as exc:
            logger.warning("Gemini batch translation failed: %s", exc)

        return [t for t in texts]

    def _build_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)

        style_hint = ""
        if target_lang == "id":
            style_hint = (
                "Use natural Indonesian that is clear and conversational, but not slangy.\n"
                "Avoid overly formal or stiff phrasing.\n"
            )

        return (
            f"You are translating {src_name} manga dialogue into natural {tgt_name}.\n"
            "Make it sound like real people speaking, not a literal translation.\n"
            "Preserve emotion, personality, and intent. Keep character names as-is.\n"
            "Use short, conversational sentences where appropriate.\n"
            f"{style_hint}"
            "Return only the translated dialogue, no extra text.\n\n"
            f"Text:\n{text}"
        )

    def _build_batch_prompt(self, texts: list[str], source_lang: str, target_lang: str) -> str:
        src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)

        style_hint = ""
        if target_lang == "id":
            style_hint = (
                "Use natural Indonesian that is clear and conversational, but not slangy.\n"
                "Avoid overly formal or stiff phrasing.\n"
            )

        numbered = "\n".join([f"{i+1}. {self._preprocess_text(t)}" for i, t in enumerate(texts)])

        return (
            f"You are translating {src_name} manga dialogue into natural {tgt_name}.\n"
            "Make it sound like real people speaking, not a literal translation.\n"
            "Preserve emotion, personality, and intent. Keep character names as-is.\n"
            "Use short, conversational sentences where appropriate.\n"
            f"{style_hint}"
            "Return only the translations as a numbered list with the same numbering.\n\n"
            f"Text:\n{numbered}"
        )

    def _call_gemini(self, prompt: str) -> str:
        url = self._build_url()
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        text = self._extract_text(data)
        if not text:
            raise ValueError("Empty response from Gemini API.")
        return text

    def _build_url(self) -> str:
        base = self.base_url.rstrip("/")
        if "/v1beta" in base or "/v1/" in base:
            url = f"{base}/{self.model}:generateContent"
        else:
            url = f"{base}/v1beta/models/{self.model}:generateContent"
        if self.api_key:
            url = f"{url}?key={self.api_key}"
        return url

    @staticmethod
    def _make_cache_key(text: str, source_lang: str, target_lang: str) -> str:
        return f"{source_lang}:{target_lang}:{text}"

    @staticmethod
    def _extract_text(data: dict) -> str:
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return ""
        return parts[0].get("text", "").strip()

    @staticmethod
    def _parse_batch_response(text: str, count: int) -> list[str]:
        if not text:
            return []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        results: list[str] = []

        for line in lines:
            if len(results) >= count:
                break
            if line[0].isdigit():
                parts = line.split(".", 1)
                if len(parts) == 2:
                    results.append(parts[1].strip())
                    continue
            results.append(line)

        if len(results) < count:
            return []

        return results[:count]

    @staticmethod
    def _preprocess_text(text: str) -> str:
        return text.replace("ï¼Ž", ".")
