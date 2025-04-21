"""Sentiment analysis helpers.

This module was split out of *analyzer.py* so the heavy logic is isolated and
can be replaced by transformer models in a future phase.
"""
from __future__ import annotations

from typing import Any, Dict, List

from spacy.tokens import Doc

try:
    from transformers import pipeline as _hf_pipeline  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    _hf_pipeline = None  # type: ignore

_DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Cache of HF pipelines keyed by model name
_PIPELINE_CACHE: dict[str, "Any"] = {}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_sentiment(analyzer: "TextAnalyzer", text: str) -> Dict[str, Any]:  # noqa: F821
    """Enhance VADER sentiment with lexicon‑aware adjustments.

    The logic is largely unchanged from the original script but now lives in a
    standalone module to allow alternate implementations.
    """
    # Decide backend ----------------------------------------------------------------
    use_transformer = analyzer.backends.sentiment == "transformer" and _hf_pipeline is not None

    doc: Doc = analyzer.nlp(text)  # type: ignore[arg-type]

    # ---------------------------------------------------------------------------
    # Transformer backend (cardiffnlp/distilbert etc.)
    # ---------------------------------------------------------------------------

    if use_transformer:
        # Lazily create pipeline and cache on the analyzer instance
        pipe = _PIPELINE_CACHE.get(_DEFAULT_MODEL)
        if pipe is None:
            pipe = _hf_pipeline("sentiment-analysis", model=_DEFAULT_MODEL, device=-1)  # CPU
            _PIPELINE_CACHE[_DEFAULT_MODEL] = pipe

        # Run on each sentence for granularity
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        hf_results = pipe(sentences, truncation=True)

        positive_elements: List[str] = []
        negative_elements: List[str] = []
        neutral_elements: List[str] = []

        pos_count = neg_count = 0
        for sent_text, res in zip(sentences, hf_results):
            label = res["label"].lower()
            score = res["score"]
            if label == "positive":
                positive_elements.append(sent_text)
                pos_count += score
            elif label == "negative":
                negative_elements.append(sent_text)
                neg_count += score
            else:
                neutral_elements.append(sent_text)

        # Aggregate overall sentiment
        if abs(pos_count - neg_count) < 1e-6:
            overall = "neutral"
        elif pos_count > neg_count:
            overall = "positive"
        else:
            overall = "negative"

        compound = (pos_count - neg_count) / max(pos_count + neg_count, 1e-6)
        enhanced_scores = {
            "pos": pos_count,
            "neg": neg_count,
            "neu": len(neutral_elements),
            "compound": compound,
        }

        return {
            "overall_sentiment": overall,
            "sentiment_scores": enhanced_scores,
            "positive_elements": positive_elements,
            "negative_elements": negative_elements,
            "neutral_elements": neutral_elements,
            "found_positive_words": [],
            "found_negative_words": [],
        }

    # ---------------------------------------------------------------------------
    # Default VADER backend (existing logic)
    # ---------------------------------------------------------------------------

    analyzer._extract_sentiment_words(doc)  # noqa: SLF001 – internal analyser method

    found_positive_words: List[str] = []
    found_negative_words: List[str] = []
    lexicon = analyzer.sentiment_analyzer.lexicon

    for word in analyzer.found_sentiment_words:
        if word in lexicon:
            score = lexicon[word]
            analyzer.word_sentiments[word] = score
            (found_positive_words if score > 0.2 else found_negative_words).append(word)

    extra_pos = {"improve", "help", "solution", "fresh", "smoother", "stronger"}
    extra_neg = {"concern", "busy", "problem", "issue", "peak"}
    for word in analyzer.found_sentiment_words:
        if word in extra_pos and word not in analyzer.word_sentiments:
            found_positive_words.append(word)
            analyzer.word_sentiments[word] = 0.4
        elif word in extra_neg and word not in analyzer.word_sentiments:
            found_negative_words.append(word)
            analyzer.word_sentiments[word] = -0.3

    base_scores = analyzer.sentiment_analyzer.polarity_scores(text)
    enhanced_scores = base_scores.copy()

    if len(found_positive_words) > len(found_negative_words) * 2:
        enhanced_scores["pos"] += 0.15
        enhanced_scores["compound"] += 0.1
    elif len(found_negative_words) > len(found_positive_words) * 2:
        enhanced_scores["neg"] += 0.15
        enhanced_scores["compound"] -= 0.1

    compound = enhanced_scores["compound"]
    if compound >= 0.05:
        overall = "positive"
    elif compound <= -0.05:
        overall = "negative"
    else:
        overall = "neutral"
    if abs(compound) < 0.2 and enhanced_scores["pos"] > 0.1 and enhanced_scores["neg"] > 0.1:
        overall = "mixed"

    positive_elements: List[str] = []
    negative_elements: List[str] = []
    neutral_elements: List[str] = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        words = {token.text.lower() for token in sent}
        pos_overlap = words.intersection(found_positive_words)
        neg_overlap = words.intersection(found_negative_words)
        sent_score = analyzer.sentiment_analyzer.polarity_scores(sent_text).copy()
        sent_score["compound"] += 0.05 * (len(pos_overlap) - len(neg_overlap))

        if sent_score["compound"] >= 0.05:
            positive_elements.append(sent_text)
        elif sent_score["compound"] <= -0.05:
            negative_elements.append(sent_text)
        else:
            neutral_elements.append(sent_text)

    return {
        "overall_sentiment": overall,
        "sentiment_scores": enhanced_scores,
        "positive_elements": list(set(positive_elements)),
        "negative_elements": list(set(negative_elements)),
        "neutral_elements": list(set(neutral_elements)),
        "found_positive_words": found_positive_words,
        "found_negative_words": found_negative_words,
    } 