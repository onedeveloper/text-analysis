# Text‑analysis

[![tests](https://github.com/your-org/text-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/text-analysis/actions)

NLP toolkit for quick extraction of key phrases and sentiment from English feedback.

## Installation (development)
```bash
# clone repo
uv venv  # create venv
uv pip install -e .[full]  # optional extras (`transformers`, `pytextrank`, etc.)
```

## CLI examples
```bash
# full analysis reading from file
uv run python -m text_analysis.cli analyse -i feedback.txt --json > out.json

# transformer sentiment only
uv run python -m text_analysis.cli sentiment "Shipping was terrible but product is great" --sentiment-backend transformer

# textrank key‑phrases with small model
uv run python -m text_analysis.cli extract -i notes.txt --model small --keyphrase-backend textrank
```

## Library usage
```python
from text_analysis.analyzer import TextAnalyzer

ana = TextAnalyzer(backends={"sentiment": "transformer", "keyphrase": "textrank"})
print(ana.extract_key_phrases("We could improve routes…"))
```

## Development
* Run tests: `uv run python -m pytest -q`
* Format with `ruff format .` and lint via `ruff check .`

---
© 2024 