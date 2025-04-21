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

from .analyzer import TextAnalyzer, analyse as pipeline_analyse

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
# Typer application
# ---------------------------------------------------------------------------


app = typer.Typer(add_completion=False, help="Analyse English text for sentiment and key phrases.")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("analyse", help="Run full analysis (key phrases + sentiment).")
def analyse(
    text: Optional[str] = typer.Argument(None, help="Text to analyse. If omitted, read STDIN."),
    input_file: Optional[Path] = typer.Option(None, "-i", exists=True, readable=True, help="Read text from file."),
    json_output: bool = typer.Option(False, "-j", help="Output JSON."),
    yaml_output: bool = typer.Option(False, "-y", help="Output YAML."),
    model: str = typer.Option("auto", help="spaCy model: auto | small | large", show_default=True, rich_help_panel="Model"),
    sentiment_backend: str = typer.Option("vader", help="Sentiment backend: vader | transformer", show_default=True, rich_help_panel="Backends"),
    keyphrase_backend: str = typer.Option("default", help="Key‑phrase backend: default | textrank", show_default=True, rich_help_panel="Backends"),
    no_colour: bool = typer.Option(False, "--no-colour", help="Disable rich colours."),
):
    _run_pipeline("analyse", text, input_file=input_file, json=json_output, yaml=yaml_output, model=model, sentiment_backend=sentiment_backend, keyphrase_backend=keyphrase_backend, no_colour=no_colour)


@app.command(help="Extract key phrases only.")
def extract(
    text: Optional[str] = typer.Argument(None, help="Text to analyse. If omitted, read STDIN."),
    input_file: Optional[Path] = typer.Option(None, "-i", exists=True, readable=True, help="Read text from file."),
    json_output: bool = typer.Option(False, "-j", help="Output JSON."),
    yaml_output: bool = typer.Option(False, "-y", help="Output YAML."),
    model: str = typer.Option("auto", help="spaCy model: auto | small | large", show_default=True, rich_help_panel="Model"),
    sentiment_backend: str = typer.Option("vader", help="Sentiment backend: vader | transformer", show_default=True, rich_help_panel="Backends"),
    keyphrase_backend: str = typer.Option("default", help="Key‑phrase backend: default | textrank", show_default=True, rich_help_panel="Backends"),
    no_colour: bool = typer.Option(False, "--no-colour", help="Disable rich colours."),
):
    _run_pipeline("extract", text, input_file=input_file, json=json_output, yaml=yaml_output, model=model, sentiment_backend=sentiment_backend, keyphrase_backend=keyphrase_backend, no_colour=no_colour)


@app.command(help="Sentiment analysis only.")
def sentiment(
    text: Optional[str] = typer.Argument(None, help="Text to analyse. If omitted, read STDIN."),
    input_file: Optional[Path] = typer.Option(None, "-i", exists=True, readable=True, help="Read text from file."),
    json_output: bool = typer.Option(False, "-j", help="Output JSON."),
    yaml_output: bool = typer.Option(False, "-y", help="Output YAML."),
    model: str = typer.Option("auto", help="spaCy model: auto | small | large", show_default=True, rich_help_panel="Model"),
    sentiment_backend: str = typer.Option("vader", help="Sentiment backend: vader | transformer", show_default=True, rich_help_panel="Backends"),
    keyphrase_backend: str = typer.Option("default", help="Key‑phrase backend: default | textrank", show_default=True, rich_help_panel="Backends"),
    no_colour: bool = typer.Option(False, "--no-colour", help="Disable rich colours."),
):
    _run_pipeline("sentiment", text, input_file=input_file, json=json_output, yaml=yaml_output, model=model, sentiment_backend=sentiment_backend, keyphrase_backend=keyphrase_backend, no_colour=no_colour)


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
        result = pipeline_analyse(text_data)
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