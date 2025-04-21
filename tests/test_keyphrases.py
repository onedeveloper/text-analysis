import pytest

from text_analysis.analyzer import TextAnalyzer


@pytest.mark.parametrize("backend", ["default", "textrank"])
def test_keyphrase_backends(backend):
    ana = TextAnalyzer(backends={"keyphrase": backend})

    if backend == "textrank":
        try:
            import pytextrank  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip("pytextrank not installed")

    text = "Drivers would like smoother routes. Peak season workload is high."
    kp = ana.extract_key_phrases(text)

    assert kp["noun_phrases"]
    assert kp["key_issues"] 