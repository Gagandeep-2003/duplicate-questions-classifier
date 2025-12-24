"""Utilities for language detection and optional translation."""
from __future__ import annotations

import logging
from typing import List, Tuple

# Optional dependency: langdetect. If unavailable, fall back to lightweight heuristics.
try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    # Make langdetect deterministic
    DetectorFactory.seed = 42
except Exception:  # pragma: no cover - handled with heuristic fallback
    DetectorFactory = None  # type: ignore
    LangDetectException = Exception  # type: ignore
    detect_langs = None

try:  # Optional dependency to avoid hard failures when translation is unavailable
    from deep_translator import GoogleTranslator
except Exception:  # pragma: no cover - optional import guard
    GoogleTranslator = None  # type: ignore

logger = logging.getLogger(__name__)


def _heuristic_language_guess(text: str) -> str | None:
    """Extremely small heuristic detector for a few common languages."""
    lowered = text.lower()
    if any(ch in lowered for ch in ["¿", "¡", "ñ", "á", "é", "í", "ó", "ú"]):
        return "es"
    if "quelle" in lowered or "français" in lowered:
        return "fr"
    if " der " in lowered or "und" in lowered:
        return "de"
    if "क्या" in lowered or "क्यों" in lowered:
        return "hi"
    if "ão" in lowered or "qual" in lowered:
        return "pt"
    return None


def detect_language_confidence(text: str, max_results: int = 3) -> List[Tuple[str, float]]:
    """Return a ranked list of language probabilities.

    Parameters
    ----------
    text: str
        Input text to analyze.
    max_results: int, default=3
        Maximum number of languages to return.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if detect_langs is not None:
        try:
            detected = detect_langs(cleaned)
            return [(d.lang, float(d.prob)) for d in detected[:max_results]]
        except LangDetectException:
            return []

    # Fallback heuristic
    guess = _heuristic_language_guess(cleaned)
    return [(guess, 0.35)] if guess else []


def primary_language(text: str) -> str | None:
    """Return the highest-probability language or ``None`` when unknown."""
    ranked = detect_language_confidence(text, max_results=1)
    return ranked[0][0] if ranked else None


def translate_to_english(text: str, source_lang: str | None = None) -> Tuple[str, bool]:
    """Translate text to English when possible.

    Returns the translated text and a flag indicating whether translation occurred.
    If translation fails or is not available, the original text is returned with a
    ``False`` flag to signal no translation.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned, False

    if GoogleTranslator is None:
        logger.warning("GoogleTranslator unavailable; skipping translation.")
        return cleaned, False

    try:
        translator = GoogleTranslator(source=source_lang or "auto", target="en")
        translated = translator.translate(cleaned)
        return translated, translated != cleaned
    except Exception as exc:  # pragma: no cover - external service variability
        logger.warning("Translation failed: %s", exc)
        return cleaned, False
