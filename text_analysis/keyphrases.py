"""Key‑phrase extraction helpers.

This module isolates the logic previously inside *TextAnalyzer.extract_key_phrases* so
it can be reused independently or swapped for alternative algorithms (e.g. TextRank).
"""
from __future__ import annotations

from typing import Any, Dict

from spacy.tokens import Token

from .config import DEFAULT_TERMS, DomainTerms

try:
    import pytextrank  # type: ignore

    _has_textrank = True
except ModuleNotFoundError:  # pragma: no cover
    _has_textrank = False


# ---------------------------------------------------------------------------
# Public entry‑point
# ---------------------------------------------------------------------------


def extract_key_phrases(
    analyzer: "TextAnalyzer",
    text: str,
    *,
    terms: DomainTerms = DEFAULT_TERMS,
) -> Dict[str, Any]:  # noqa: F821
    """Return noun/action/sentiment phrases and key issues.

    Parameters
    ----------
    analyzer
        Instance of :class:`text_analysis.analyzer.TextAnalyzer` that already owns
        a spaCy pipeline, sentiment_words, etc.
    text
        Raw text to analyse.
    terms
        Domain‑specific term buckets that define *key issues*.
    """
    # Use TextRank backend if requested and available --------------------------------
    if analyzer.backends.keyphrase == "textrank":
        if not _has_textrank:
            raise RuntimeError("pytextrank not installed; install with `uv add pytextrank`. ")

        # Ensure TextRank is added once to pipeline
        if "textrank" not in analyzer.nlp.pipe_names:
            analyzer.nlp.add_pipe("textrank")

        doc = analyzer.nlp(text)
        ranked_phrases = [phrase.text for phrase in doc._.phrases[:20]]
        # For compatibility, return only noun phrases as ranked words for now
        noun_phrases = ranked_phrases
        action_phrases: list[str] = []  # no verb extraction when using textrank
    else:
        doc = analyzer.nlp(text)
        action_phrases: list[str] = []

    if analyzer.backends.keyphrase != "textrank":
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    # Verb‑centric phrases --------------------------------------------------
    for sent in doc.sents:
        for token in sent:  # type: Token
            if token.pos_ == "VERB":
                phrase = _get_verb_phrase(token)
                if phrase and len(phrase.split()) > 2:
                    action_phrases.append(phrase)

    # Sentiment phrases / dynamic sentiment lexicon ------------------------
    sentiment_phrases: list[str] = []
    dynamic_sentiment_words = analyzer._extract_sentiment_words(doc)  # noqa: SLF001 – internal use is fine here
    analyzer.sentiment_words.update(dynamic_sentiment_words)
    for sent in doc.sents:
        if any(word in sent.text.lower() for word in analyzer.sentiment_words):
            sentiment_phrases.append(sent.text.strip())

    # Heuristic key‑issues --------------------------------------------------
    issues: list[str] = []

    def _match(sentence_terms: list[str], label: str) -> None:
        matched = [s.text for s in doc.sents if any(t in s.text.lower() for t in sentence_terms)]
        if matched:
            issues.append(f"{label}: {matched[0]}")

    _match(terms.suggestions, "Suggestion for improvement")
    _match(terms.business, "Business concern")
    _match(terms.workload, "Workload perception")
    _match(terms.infrastructure, "Infrastructure recommendation")

    if not issues:
        issues = analyzer._extract_fallback_issues(text)  # noqa: SLF001

    return {
        "noun_phrases": list(set(noun_phrases)),
        "action_phrases": list(set(action_phrases)),
        "sentiment_phrases": list(set(sentiment_phrases)),
        "key_issues": issues,
    }


# ---------------------------------------------------------------------------
# Internal helpers (unchanged from original)
# ---------------------------------------------------------------------------


def _get_verb_phrase(verb_token):  # noqa: ANN001 – spaCy Token type
    phrase_tokens = [verb_token]
    phrase_tokens.extend(c for s in verb_token.children if s.dep_ in ("nsubj", "nsubjpass") for c in s.subtree)
    phrase_tokens.extend(c for o in verb_token.children if o.dep_ in ("dobj", "pobj", "iobj") for c in o.subtree)
    phrase_tokens = sorted(set(phrase_tokens), key=lambda t: t.i)
    return " ".join(t.text for t in phrase_tokens) 