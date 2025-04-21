import json

from text_analysis.analyzer import analyse


def test_analyse_basic():
    text = "We should improve routes. Drivers are busy during peak season."
    result = analyse(text)

    assert result["sentiment"]["overall_sentiment"] in {"positive", "neutral", "negative", "mixed"}
    assert "noun_phrases" in result["key_phrases"]
    assert result["key_phrases"]["key_issues"]

    # JSON serialisable
    json.dumps(result) 