import logging
import os
import asyncio
from typing import Optional, List

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# Check if aiohttp is available
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not installed, async translation disabled")

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

# Language-specific style hints for natural, localized translations
# Based on professional localization guidelines (Netflix, Totus, Unbabel)
STYLE_HINTS = {
    "id": (
        "Gunakan bahasa Indonesia baku yang tetap luwes dan enak dibaca.\n"
        "Gunakan pronomina netral secara default: aku untuk 'I', kamu atau kau untuk 'you'.\n"
        "Jangan gunakan gaya slang keras seperti 'gue/lo'.\n"
        "Jangan gunakan partikel percakapan seperti: ya, kan, deh, dong, sih, kok.\n"
        "Hindari singkatan tidak baku (mis. yg, gak, dll). Gunakan 'tidak' atau 'tak'.\n"
        "Gunakan 'saya/Anda' hanya jika konteks benar-benar formal atau berjarak (mis. rapat, layanan resmi, berbicara dengan atasan/klien, atau orang yang sangat dihormati).\n"
        "Utamakan lokalisasi daripada terjemahan harfiah; pertahankan emosi dan maksud.\n"
        "Jaga dialog ringkas, jelas, dan cocok untuk ruang balon manga.\n"
        "Pertahankan nama diri, nama tempat, dan istilah khusus apa adanya.\n"
    ),

    "en": (
        "Use standard American English.\n"
        "Localize rather than translate literally; preserve tone, emotion, and intent.\n"
        "Keep dialogue concise and natural for manga bubble space.\n"
        "Avoid internet slang and abbreviations (e.g., LOL, WTF, btw).\n"
        "Use contractions when natural (e.g., I'm, don't) unless the character is formal.\n"
        "Sound effects should be single impactful words in all caps (e.g., 'BANG', 'CRASH').\n"
        "Keep proper names as-is.\n"
    ),

    "es": (
        "Usa español neutro; evita regionalismos fuertes salvo que el contexto lo exija.\n"
        "Localiza en lugar de traducir literalmente; conserva tono, emoción e intención.\n"
        "Mantén el diálogo breve y natural para el espacio de los bocadillos.\n"
        "Elige el registro adecuado (tú/usted) según la relación entre personajes.\n"
        "Usa correctamente los signos de apertura (¿ ¡).\n"
        "Mantén los nombres propios sin cambios.\n"
    ),

    "fr": (
        "Utilise un français standard.\n"
        "Localise plutôt que traduire mot à mot; conserve le ton, l'émotion et l'intention.\n"
        "Garde des répliques courtes et naturelles, adaptées aux bulles.\n"
        "Choisis le registre approprié (tu/vous) selon la relation entre personnages.\n"
        "Respecte la typographie française : espace avant ? ! : ;.\n"
        "Évite les phrases trop longues et les formulations lourdes.\n"
        "Conserve les noms propres tels quels.\n"
    ),

    "de": (
        "Verwende Standarddeutsch (Hochdeutsch).\n"
        "Lokalisieren statt wortwörtlich übersetzen; Ton, Emotion und Absicht beibehalten.\n"
        "Halte Dialoge knapp und natürlich für Sprechblasen.\n"
        "Wähle die passende Anrede (du/Sie) je nach Beziehung und Situation.\n"
        "Achte auf gut lesbare zusammengesetzte Wörter; vermeide unnötige Bandwurmsätze.\n"
        "Eigennamen unverändert lassen.\n"
    ),

    "pt": (
        "Use Português do Brasil (pt-BR) por padrão, salvo instrução contrária.\n"
        "Localize em vez de traduzir ao pé da letra; preserve tom, emoção e intenção.\n"
        "Mantenha o diálogo curto e natural para balões.\n"
        "Escolha o registro adequado (você/tu; formalidade) conforme a relação entre personagens.\n"
        "Evite construções excessivamente formais em cenas casuais.\n"
        "Mantenha nomes próprios como estão.\n"
    ),

    "ru": (
        "Используй естественный русский язык.\n"
        "Локализуй, а не переводи дословно; сохраняй тон, эмоции и намерение.\n"
        "Делай реплики короткими и удобными для облачков.\n"
        "Выбирай уровень вежливости (ты/вы) по контексту и отношениям персонажей.\n"
        "Избегай канцелярита и слишком книжных оборотов в бытовых сценах.\n"
        "Собственные имена оставляй без изменений.\n"
    ),

    "vi": (
        "Dùng tiếng Việt tự nhiên.\n"
        "Ưu tiên bản địa hoá hơn dịch từng chữ; giữ đúng sắc thái cảm xúc và ý định.\n"
        "Lời thoại ngắn gọn, phù hợp khung thoại manga.\n"
        "Chọn đại từ xưng hô theo tuổi tác và quan hệ (anh/em, chị/em, tôi/bạn, v.v.).\n"
        "Tránh văn phong quá trang trọng hoặc quá văn chương trong cảnh đời thường.\n"
        "Giữ nguyên tên riêng.\n"
    ),

    "th": (
        "ใช้ภาษาไทยที่เป็นธรรมชาติและอ่านลื่นไหล.\n"
        "เน้นการปรับให้เข้ากับภาษาไทยมากกว่าการแปลตรงตัว โดยคงโทน อารมณ์ และเจตนาเดิม.\n"
        "ทำบทสนทนาให้สั้น กระชับ เหมาะกับพื้นที่บอลลูน.\n"
        "เลือกระดับความสุภาพให้เหมาะกับความสัมพันธ์และสถานการณ์ และใช้คำลงท้าย (ครับ/ค่ะ) เฉพาะเมื่อเหมาะสม.\n"
        "หลีกเลี่ยงสำนวนทางการเกินไปในฉากสบาย ๆ.\n"
        "คงชื่อเฉพาะไว้ตามเดิม.\n"
    ),

    "ms": (
        "Gunakan Bahasa Melayu standard.\n"
        "Utamakan pelokalan berbanding terjemahan literal; kekalkan nada, emosi, dan niat.\n"
        "Pastikan dialog ringkas dan sesuai untuk ruang gelembung manga.\n"
        "Pilih ganti nama dan tahap kesantunan mengikut hubungan watak dan situasi.\n"
        "Elakkan bahasa terlalu rasmi dalam babak santai.\n"
        "Kekalkan nama khas seperti asal.\n"
    ),

    "ko": (
        "자연스러운 한국어로 번역하세요.\n"
        "직역보다 현지화를 우선하며, 톤/감정/의도를 유지하세요.\n"
        "말풍선에 맞게 대사는 간결하게 유지하세요.\n"
        "인물 관계에 따라 말투를 선택하세요(반말/존댓말, 해요체/하십시오체 등).\n"
        "일상 장면에서 과도하게 문어체/격식을 차린 표현을 피하세요.\n"
        "고유명사는 그대로 유지하세요.\n"
    ),

    "zh": (
        "使用自然的简体中文。\n"
        "优先本地化而非逐字直译；保留语气、情绪与意图。\n"
        "对话要简洁，适合漫画气泡框。\n"
        "根据人物关系选择合适的称呼与语气（亲密/礼貌/疏离）。\n"
        "避免过于书面或生硬的机器翻译腔。\n"
        "专有名词与人名保持不变。\n"
    ),

    "zh-tw": (
        "使用自然的繁體中文。\n"
        "優先在地化而非逐字直譯；保留語氣、情緒與意圖。\n"
        "對話要精簡，適合漫畫對話框。\n"
        "依人物關係選擇合適稱呼與語氣（親密/禮貌/疏離）。\n"
        "避免過度書面或生硬的機器翻譯語感。\n"
        "專有名詞與人名保持不變。\n"
    ),
}

DEFAULT_STYLE_HINT = (
    "Localize rather than translate literally.\n"
    "Preserve tone, emotion, and intent; keep character names as-is.\n"
    "Keep dialogue concise and suitable for manga bubble space.\n"
    "Use natural phrasing that sounds like real speech in the target language.\n"
    "Choose formality/register based on character relationships and the scene.\n"
)



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

        # Get language-specific style hint or default
        style_hint = STYLE_HINTS.get(target_lang, DEFAULT_STYLE_HINT)

        return (
            f"You are a professional manga translator, translating {src_name} manga dialogue into natural {tgt_name}.\n"
            "Make it sound like real people speaking, not a literal translation.\n"
            "Preserve emotion, personality, character voice, and intent.\n"
            "Keep character names and proper nouns as-is (do not translate names).\n"
            "Use short, conversational sentences that fit in speech bubbles.\n"
            f"{style_hint}"
            "Return only the translated dialogue, no commentary or extra text.\n\n"
            f"Text:\n{text}"
        )

    def _build_batch_prompt(self, texts: list[str], source_lang: str, target_lang: str) -> str:
        src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)

        # Get language-specific style hint or default
        style_hint = STYLE_HINTS.get(target_lang, DEFAULT_STYLE_HINT)

        numbered = "\n".join([f"{i+1}. {self._preprocess_text(t)}" for i, t in enumerate(texts)])

        return (
            f"You are a professional manga translator, translating {src_name} manga dialogue into natural {tgt_name}.\n"
            "Make it sound like real people speaking, not a literal translation.\n"
            "Preserve emotion, personality, character voice, and intent.\n"
            "Keep character names and proper nouns as-is (do not translate names).\n"
            "Use short, conversational sentences that fit in speech bubbles.\n"
            f"{style_hint}"
            "Return only the translations as a numbered list matching the input numbering.\n"
            "Each line should contain only the translated text for that item.\n\n"
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


# =============================================================================
# Async Translator for Parallel Processing
# =============================================================================

class AsyncTranslator:
    """
    Async translator for batch processing with concurrent API requests.
    Uses aiohttp for non-blocking HTTP calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_url: Optional[str] = None,
        max_concurrent: int = 4,
        timeout_seconds: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.base_url = api_url or os.getenv("GEMINI_API_URL", DEFAULT_API_URL)
        self.timeout = timeout_seconds
        self.max_concurrent = max_concurrent
        self._cache: dict[str, str] = {}

    def _build_url(self) -> str:
        base = self.base_url.rstrip("/")
        if "/v1beta" in base or "/v1/" in base:
            url = f"{base}/{self.model}:generateContent"
        else:
            url = f"{base}/v1beta/models/{self.model}:generateContent"
        if self.api_key:
            url = f"{url}?key={self.api_key}"
        return url

    def _build_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        style_hint = STYLE_HINTS.get(target_lang, DEFAULT_STYLE_HINT)

        return (
            f"You are a professional manga translator, translating {src_name} manga dialogue into natural {tgt_name}.\n"
            "Make it sound like real people speaking, not a literal translation.\n"
            "Preserve emotion, personality, character voice, and intent.\n"
            "Keep character names and proper nouns as-is (do not translate names).\n"
            "Use short, conversational sentences that fit in speech bubbles.\n"
            f"{style_hint}"
            "Return only the translated dialogue, no commentary or extra text.\n\n"
            f"Text:\n{text}"
        )

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
    def _make_cache_key(text: str, source_lang: str, target_lang: str) -> str:
        return f"{source_lang}:{target_lang}:{text}"

    async def translate_one_async(
        self,
        session: "aiohttp.ClientSession",
        text: str,
        source_lang: str,
        target_lang: str,
        semaphore: asyncio.Semaphore,
        retry_count: int = 2,
    ) -> str:
        """Translate single text asynchronously with retry support."""
        if not text or not text.strip():
            return ""

        # Check cache first
        cache_key = self._make_cache_key(text, source_lang, target_lang)
        if cache_key in self._cache:
            return self._cache[cache_key]

        text_preview = text[:40].replace('\n', ' ')
        last_error = None

        for attempt in range(retry_count + 1):
            async with semaphore:
                prompt = self._build_prompt(text, source_lang, target_lang)
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
                }
                url = self._build_url()

                try:
                    # Increase timeout for retries
                    timeout = self.timeout * (1 + attempt * 0.5)
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            result = self._extract_text(data)
                            if result:
                                self._cache[cache_key] = result
                                return result
                            else:
                                logger.warning(f"Empty response for: {text_preview}...")
                                return text
                        elif response.status == 429:
                            # Rate limited - wait and retry
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                            await asyncio.sleep(wait_time)
                            last_error = f"Rate limited (429)"
                            continue
                        elif response.status >= 500:
                            # Server error - retry
                            wait_time = 1 * (attempt + 1)
                            logger.warning(f"Server error {response.status}, retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            last_error = f"Server error ({response.status})"
                            continue
                        else:
                            logger.warning(f"Gemini API error {response.status} for: {text_preview}...")
                            return text

                except asyncio.TimeoutError:
                    last_error = "timeout"
                    if attempt < retry_count:
                        logger.warning(f"Translation timeout (attempt {attempt + 1}/{retry_count + 1}) for: {text_preview}...")
                        await asyncio.sleep(1)
                        continue
                    else:
                        logger.warning(f"Translation timeout for: {text_preview}...")
                        return text

                except aiohttp.ClientError as e:
                    last_error = str(e)
                    if attempt < retry_count:
                        logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
                        await asyncio.sleep(1)
                        continue
                    else:
                        logger.warning(f"Translation connection failed for: {text_preview}... - {e}")
                        return text

                except Exception as e:
                    logger.warning(f"Translation failed for: {text_preview}... - {e}")
                    return text

        # All retries exhausted
        logger.warning(f"Translation failed after {retry_count + 1} attempts for: {text_preview}... (last error: {last_error})")
        return text

    async def translate_batch_async(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """
        Translate multiple texts concurrently.
        Uses semaphore to limit concurrent requests.
        """
        if not AIOHTTP_AVAILABLE:
            # Fallback to sync translator
            sync_translator = MangaTranslator(
                api_key=self.api_key,
                model=self.model,
                api_url=self.base_url,
            )
            return sync_translator.translate_batch(texts, source_lang, target_lang)

        if not texts:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async with aiohttp.ClientSession() as session:
            tasks = [
                self.translate_one_async(session, text, source_lang, target_lang, semaphore)
                for text in texts
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Translation {i} failed: {result}")
                final_results.append(texts[i])  # Return original on error
            else:
                final_results.append(result)

        return final_results


# Global async translator instance
_async_translator: Optional[AsyncTranslator] = None


def get_async_translator() -> AsyncTranslator:
    """Get or create async translator singleton."""
    global _async_translator
    if _async_translator is None:
        _async_translator = AsyncTranslator()
    return _async_translator


async def translate_batch_async(
    texts: List[str],
    source_lang: str,
    target_lang: str,
) -> List[str]:
    """Convenience function for batch translation."""
    translator = get_async_translator()
    return await translator.translate_batch_async(texts, source_lang, target_lang)
