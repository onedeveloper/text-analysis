"""Rich Typer CLI for *text_analysis* (Phase 3).

Commands
--------
analyse   – full pipeline (key phrases + sentiment)
extract   – key‑phrase extraction only
sentiment – sentiment analysis only

Common options:
  •  `--input-file/-i PATH`  read text from file
  •  `TEXT` argument         read text directly or, if omitted, STDIN
  •  `--json/--yaml`         choose structured output format
  •  `--model`               spaCy model: auto | small | large  (default: auto)
  •  `--sentiment-backend`   vader | transformer
  •  `--keyphrase-backend`   default | textrank
  •  `--no-colour`           disable rich colours
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print  # type: ignore

try:
    import yaml  # PyYAML

    _has_yaml = True
except ModuleNotFoundError:  # pragma: no cover
    _has_yaml = False

from .analyzer import TextAnalyzer, analyse

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _read_text_arg(text: Optional[str], input_file: Optional[Path]) -> str:
    if input_file is not None:
        return input_file.read_text()
    if text is None:
        return sys.stdin.read()
    return text


def _print_structured(result, *, as_json: bool, as_yaml: bool, no_colour: bool = False):
    if as_json:
        print(json.dumps(result, indent=2))
    elif as_yaml:
        if not _has_yaml:
            raise RuntimeError("PyYAML not installed – run `uv add pyyaml`.")
        print(yaml.safe_dump(result, sort_keys=False))
    else:
        from pprint import pprint

        pprint(result)


# ---------------------------------------------------------------------------
# Typer application & common options
# ---------------------------------------------------------------------------


app = typer.Typer(add_completion=False, help="Analyse English text for sentiment and key phrases.")


def common_options(f):  # decorator to share options between commands
    f = typer.option("--input-file", "-i", exists=True, readable=True, help="Read text from file.")(f)
    f = typer.option("--json", "-j", is_flag=True, help="Output JSON.")(f)
    f = typer.option("--yaml", "-y", is_flag=True, help="Output YAML.")(f)
    f = typer.option(
        "--model",
        help="spaCy model: auto | small | large",
        default="auto",
        show_default=True,
        rich_help_panel="Model",
    )(f)
    f = typer.option(
        "--sentiment-backend",
        help="Sentiment backend: vader | transformer",
        default="vader",
        show_default=True,
        rich_help_panel="Backends",
    )(f)
    f = typer.option(
        "--keyphrase-backend",
        help="Key‑phrase backend: default | textrank",
        default="default",
        show_default=True,
        rich_help_panel="Backends",
    )(f)
    f = typer.option("--no-colour", is_flag=True, help="Disable rich colours.")(f)
    return f


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command(help="Run full analysis (key phrases + sentiment).")
@common_options
def analyse_cmd(
    text: Optional[str] = typer.Argument(None, help="Text to analyse. If omitted, read STDIN."),
    **kwargs,
):
    _run_pipeline("analyse", text, **kwargs)


@app.command(help="Extract key phrases only.")
@common_options
def extract(
    text: Optional[str] = typer.Argument(None, help="Text to analyse. If omitted, read STDIN."),
    **kwargs,
):
    _run_pipeline("extract", text, **kwargs)


@app.command(help="Sentiment analysis only.")
@common_options
def sentiment(
    text: Optional[str] = typer.Argument(None, help="Text to analyse. If omitted, read STDIN."),
    **kwargs,
):
    _run_pipeline("sentiment", text, **kwargs)


# ---------------------------------------------------------------------------
# Implementation details
# ---------------------------------------------------------------------------


def _run_pipeline(command: str, text: Optional[str], **opts):  # noqa: C901 – acceptable for CLI
    text_data = _read_text_arg(text, opts.pop("input_file"))

    # Map model choice -----------------------------------------------------
    model_arg = opts.pop("model")
    model_map = {
        "small": "en_core_web_sm",
        "large": "en_core_web_trf",
        "auto": "auto",
    }
    model_name = model_map.get(model_arg, "auto")

    backends = {
        "sentiment": opts.pop("sentiment_backend"),
        "keyphrase": opts.pop("keyphrase_backend"),
    }

    # Initialise analyzer (model download progress is shown by spaCy)
    analyzer = TextAnalyzer(model=model_name, backends=backends)

    # Execute --------------------------------------------------------------
    if command == "analyse":
        result = analyse(text_data)
    elif command == "extract":
        result = analyzer.extract_key_phrases(text_data)
    elif command == "sentiment":
        result = analyzer.analyze_sentiment(text_data)
    else:  # pragma: no cover – should not happen
        raise RuntimeError(command)

    _print_structured(result, as_json=opts.pop("json"), as_yaml=opts.pop("yaml"), no_colour=opts.pop("no_colour"))


def _main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    _main() 