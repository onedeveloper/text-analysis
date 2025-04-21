import pytest

from text_analysis.analyzer import TextAnalyzer


@pytest.mark.parametrize("backend", ["vader", "transformer"])
def test_sentiment_backends(backend):
    ana = TextAnalyzer(backends={"sentiment": backend})

    if backend == "transformer":
        try:
            import transformers  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip("transformers not installed")

    res = ana.analyze_sentiment("I love the fresh routes but they are sometimes busy and difficult.")
    assert res["overall_sentiment"] in {"positive", "negative", "neutral", "mixed"}
    assert res["sentiment_scores"] 