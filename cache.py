"""
Translation cache with in-memory LRU and optional SQLite persistence.
Also includes V2 image cache for inpainted results.
"""

import hashlib
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)


class TranslationCache:
    """Two-tier cache: Memory LRU + optional SQLite persistent."""

    def __init__(self, max_memory_items: int = 1000, db_path: Optional[str] = None):
        self._memory_cache: OrderedDict[str, str] = OrderedDict()
        self._max_memory = max_memory_items
        self._lock = threading.Lock()
        self._db_path = db_path
        self._db_conn: Optional[sqlite3.Connection] = None

        if db_path:
            self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for persistent cache."""
        try:
            self._db_conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._db_conn.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    key TEXT PRIMARY KEY,
                    src_text TEXT,
                    tgt_text TEXT,
                    src_lang TEXT,
                    tgt_lang TEXT,
                    created_at REAL
                )
            """)
            self._db_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created ON translations(created_at)"
            )
            self._db_conn.commit()
            logger.info(f"Translation cache DB initialized: {self._db_path}")
        except Exception as e:
            logger.warning(f"Failed to init cache DB: {e}")
            self._db_conn = None

    @staticmethod
    def _make_key(text: str, src_lang: str, tgt_lang: str) -> str:
        """Create cache key from text and language pair."""
        raw = f"{src_lang}:{tgt_lang}:{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, text: str, src_lang: str, tgt_lang: str) -> Optional[str]:
        """Get cached translation."""
        key = self._make_key(text, src_lang, tgt_lang)

        with self._lock:
            # Check memory first
            if key in self._memory_cache:
                self._memory_cache.move_to_end(key)
                return self._memory_cache[key]

            # Check disk
            if self._db_conn:
                try:
                    cursor = self._db_conn.execute(
                        "SELECT tgt_text FROM translations WHERE key = ?", (key,)
                    )
                    row = cursor.fetchone()
                    if row:
                        # Promote to memory
                        self._memory_cache[key] = row[0]
                        self._enforce_memory_limit()
                        return row[0]
                except Exception as e:
                    logger.warning(f"Cache DB read error: {e}")

        return None

    def set(self, text: str, translation: str, src_lang: str, tgt_lang: str) -> None:
        """Store translation in cache."""
        key = self._make_key(text, src_lang, tgt_lang)

        with self._lock:
            self._memory_cache[key] = translation
            self._memory_cache.move_to_end(key)
            self._enforce_memory_limit()

            # Write to disk
            if self._db_conn:
                try:
                    self._db_conn.execute(
                        """INSERT OR REPLACE INTO translations
                           (key, src_text, tgt_text, src_lang, tgt_lang, created_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (key, text, translation, src_lang, tgt_lang, time.time())
                    )
                    self._db_conn.commit()
                except Exception as e:
                    logger.warning(f"Cache DB write error: {e}")

    def get_batch(
        self, texts: List[str], src_lang: str, tgt_lang: str
    ) -> Tuple[Dict[int, str], List[str], List[int]]:
        """
        Get cached translations for batch.

        Returns:
            (cached_results, uncached_texts, uncached_indices)
        """
        cached: Dict[int, str] = {}
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []

        for i, text in enumerate(texts):
            result = self.get(text, src_lang, tgt_lang)
            if result is not None:
                cached[i] = result
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        return cached, uncached_texts, uncached_indices

    def _enforce_memory_limit(self) -> None:
        """Evict oldest entries if over limit."""
        while len(self._memory_cache) > self._max_memory:
            self._memory_cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cache."""
        with self._lock:
            self._memory_cache.clear()
            if self._db_conn:
                self._db_conn.execute("DELETE FROM translations")
                self._db_conn.commit()

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        memory_count = len(self._memory_cache)
        db_count = 0

        if self._db_conn:
            try:
                cursor = self._db_conn.execute("SELECT COUNT(*) FROM translations")
                db_count = cursor.fetchone()[0]
            except Exception:
                pass

        return {
            "memory_items": memory_count,
            "db_items": db_count,
            "max_memory": self._max_memory
        }


# Singleton instance
_cache_instance: Optional[TranslationCache] = None


def get_translation_cache(
    max_memory: int = 1000, db_path: Optional[str] = None
) -> TranslationCache:
    """Get singleton cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = TranslationCache(max_memory, db_path)
    return _cache_instance


# =============================================================================
# V2 Image Cache - For inpainted results
# =============================================================================

class V2ImageCache:
    """
    Cache for V2 API inpainted images.
    Stores base64-encoded images with their translation regions.
    Uses memory LRU with configurable size limit.
    """

    def __init__(self, max_items: int = 100, max_size_mb: int = 500):
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._max_items = max_items
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._current_size = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(image_hash: str, mode: str, source_lang: str, target_lang: str,
                 detect_osb: bool = False, inpaint_method: str = "auto") -> str:
        """Create unique cache key from request parameters."""
        raw = f"{image_hash}:{mode}:{source_lang}:{target_lang}:{detect_osb}:{inpaint_method}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, key: str) -> Optional[Dict]:
        """
        Get cached V2 result.

        Returns:
            Dict with 'image_b64', 'regions', 'meta' or None if not cached.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                cached = self._cache[key]
                cached['meta']['cache_hit'] = True
                return cached
            self._misses += 1
            return None

    def set(self, key: str, image_b64: Optional[str], regions: List[Dict],
            meta: Dict) -> None:
        """
        Store V2 result in cache.

        Args:
            key: Cache key from make_key()
            image_b64: Base64-encoded inpainted image (can be None for realtime)
            regions: List of region dictionaries
            meta: Metadata dictionary
        """
        with self._lock:
            # Calculate size of this entry
            entry_size = 0
            if image_b64:
                entry_size = len(image_b64)

            # Check if we need to evict entries
            while (len(self._cache) >= self._max_items or
                   self._current_size + entry_size > self._max_size_bytes):
                if not self._cache:
                    break
                # Evict oldest entry
                old_key, old_entry = self._cache.popitem(last=False)
                if old_entry.get('image_b64'):
                    self._current_size -= len(old_entry['image_b64'])

            # Store new entry
            self._cache[key] = {
                'image_b64': image_b64,
                'regions': regions,
                'meta': {**meta, 'cached_at': time.time()},
            }
            self._current_size += entry_size

    def invalidate(self, image_hash: str) -> int:
        """
        Invalidate all cache entries for a specific image.

        Args:
            image_hash: The image hash to invalidate

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(image_hash[:16])]
            for key in keys_to_remove:
                entry = self._cache.pop(key, None)
                if entry and entry.get('image_b64'):
                    self._current_size -= len(entry['image_b64'])
            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            hit_rate = 0.0
            total = self._hits + self._misses
            if total > 0:
                hit_rate = self._hits / total

            return {
                'items': len(self._cache),
                'max_items': self._max_items,
                'size_mb': round(self._current_size / (1024 * 1024), 2),
                'max_size_mb': self._max_size_bytes // (1024 * 1024),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(hit_rate, 3),
            }


# Singleton instance for V2 cache
_v2_cache_instance: Optional[V2ImageCache] = None


def get_v2_image_cache(max_items: int = 100, max_size_mb: int = 500) -> V2ImageCache:
    """Get singleton V2 image cache instance."""
    global _v2_cache_instance
    if _v2_cache_instance is None:
        _v2_cache_instance = V2ImageCache(max_items, max_size_mb)
    return _v2_cache_instance
