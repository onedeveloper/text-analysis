"""Core analyser module housing :class:`TextAnalyzer` and helper :func:`analyse`.

This code is largely ported from the original *text_analysis.py* script so that it can
be imported as a library. Heavy refactors (dataclasses, new models) are planned for a
future phase but the public interface should remain stable.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict

import nltk
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from .config import DEFAULT_THRESHOLDS, Thresholds, DEFAULT_BACKENDS, Backends
import spacy.util
from spacy.cli import download as spacy_download
from .keyphrases import extract_key_phrases as _extract_key_phrases
from .sentiment import analyze_sentiment as _analyze_sentiment

# Lazy download of NLTK resources (silent)
for resource in ("punkt", "vader_lexicon", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:  # pragma: no cover – runs once per machine
        nltk.download(resource, quiet=True)

# Process‑level spaCy pipeline cache
_NLP_CACHE: dict[str, spacy.language.Language] = {}


class TextAnalyzer:
    """Combine key‑phrase extraction and sentiment analysis."""

    __slots__ = ("nlp", "sentiment_analyzer", "stopwords", "thresholds", "backends", "sentiment_words", "found_sentiment_words", "word_sentiments", "sentiment_threshold")

    def __init__(
        self,
        model: str = "auto",
        *,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
        backends: Backends = DEFAULT_BACKENDS,
    ) -> None:
        # Resolve "auto" model choice -------------------------------------
        if model == "auto":
            if spacy.prefer_gpu():
                model = "en_core_web_trf"
            else:
                model = "en_core_web_sm"

        # Load selected spaCy model ---------------------------------------
        if model in _NLP_CACHE:
            self.nlp = _NLP_CACHE[model]
        else:
            # Ensure model is installed once (persistent cache)
            if not spacy.util.is_package(model):
                print(f"[text-analysis] Downloading spaCy model {model!r}…", file=sys.stderr)
                spacy_download(model, direct=True, quiet=True)
            self.nlp = spacy.load(model)  # type: ignore[assignment]
            _NLP_CACHE[model] = self.nlp

        # Sentiment analyser (VADER for now).
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Stopword set (cached for performance).
        self.stopwords = set(nltk.corpus.stopwords.words("english"))

        # Sentiment word tracking
        self.thresholds = thresholds
        self.sentiment_threshold = thresholds.sentiment
        self.sentiment_words = self._initialize_sentiment_words()
        self.found_sentiment_words: set[str] = set()
        self.word_sentiments: Dict[str, float] = {}
        # Coerce mapping into Backends dataclass for convenience
        if isinstance(backends, dict):
            backends = Backends(**backends)  # type: ignore[arg-type]
        self.backends = backends

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def extract_key_phrases(self, text: str):  # noqa: D401
        """Delegate to :pymeth:`text_analysis.keyphrases.extract_key_phrases`."""
        return _extract_key_phrases(self, text)

    def analyze_sentiment(self, text: str):  # noqa: D401
        """Delegate to sentiment analyzer module."""
        return _analyze_sentiment(self, text)

    # ------------------------------------------------------------------
    # Internal helpers (copied / cleaned from original script)
    # ------------------------------------------------------------------

    def _get_verb_phrase(self, verb_token):  # noqa: ANN001 (spaCy token)
        phrase_tokens = [verb_token]
        phrase_tokens.extend(c for s in verb_token.children if s.dep_ in ("nsubj", "nsubjpass") for c in s.subtree)
        phrase_tokens.extend(c for o in verb_token.children if o.dep_ in ("dobj", "pobj", "iobj") for c in o.subtree)
        phrase_tokens = sorted(set(phrase_tokens), key=lambda t: t.i)
        return " ".join(t.text for t in phrase_tokens)

    def _initialize_sentiment_words(self):  # noqa: D401 – kept from original
        baseline = {
            "feeling",
            "help",
            "should",
            "busy",
            "concern",
            "improve",
            "problem",
            "issue",
            "excellent",
            "terrible",
            "frustrat",
            "worry",
            "difficult",
            "solution",
        }
        lexicon = self.sentiment_analyzer.lexicon
        strong = {
            w
            for w, score in lexicon.items()
            if abs(score) > self.sentiment_threshold and len(w) > 3 and "_" not in w
        }
        return baseline.union(strong)

    def _extract_sentiment_words(self, doc):  # noqa: D401
        sentiment_words: set[str] = set()
        self.found_sentiment_words = set()
        for token in doc:
            if token.is_stop or token.is_punct or len(token.text) < 4:
                continue
            if token.pos_ not in ("ADJ", "ADV", "VERB", "NOUN"):
                continue
            word = token.text.lower()
            score = self.sentiment_analyzer.polarity_scores(token.text)["compound"]
            if abs(score) > self.sentiment_threshold:
                sentiment_words.add(word)
            if word in self.sentiment_words:
                self.found_sentiment_words.add(word)
        return sentiment_words

    @staticmethod
    def _extract_fallback_issues(text: str):
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        return [f"Issue {i+1}: {s}" for i, s in enumerate(sentences)]


# ----------------------------------------------------------------------
# Helper facade
# ----------------------------------------------------------------------

def analyse(text: str) -> Dict[str, Any]:
    """Quick one‑shot analysis returning *key_phrases* and *sentiment*."""
    ana = TextAnalyzer()
    return {
        "key_phrases": ana.extract_key_phrases(text),
        "sentiment": ana.analyze_sentiment(text),
    }


# ----------------------------------------------------------------------
# Legacy CLI entry‑point (kept for backwards compatibility)
# ----------------------------------------------------------------------

def _main_legacy() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Analyse text from CLI (legacy entry‑point).")
    parser.add_argument("text", nargs="?", help="Text to analyse. If not provided, read stdin.")
    parser.add_argument("--json", action="store_true", help="Return JSON and exit.")
    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    result = analyse(text)
    if args.json:
        import json

        print(json.dumps(result, indent=2))
    else:
        from pprint import pprint

        pprint(result)


if __name__ == "__main__":  # pragma: no cover
    _main_legacy() 