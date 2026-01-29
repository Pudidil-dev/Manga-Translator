"""
Translation cache with in-memory LRU and optional SQLite persistence.
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

    def _init_db(self):
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

    def set(self, text: str, translation: str, src_lang: str, tgt_lang: str):
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

    def _enforce_memory_limit(self):
        """Evict oldest entries if over limit."""
        while len(self._memory_cache) > self._max_memory:
            self._memory_cache.popitem(last=False)

    def clear(self):
        """Clear all cache."""
        with self._lock:
            self._memory_cache.clear()
            if self._db_conn:
                self._db_conn.execute("DELETE FROM translations")
                self._db_conn.commit()

    def stats(self) -> dict:
        """Get cache statistics."""
        memory_count = len(self._memory_cache)
        db_count = 0

        if self._db_conn:
            try:
                cursor = self._db_conn.execute("SELECT COUNT(*) FROM translations")
                db_count = cursor.fetchone()[0]
            except:
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
