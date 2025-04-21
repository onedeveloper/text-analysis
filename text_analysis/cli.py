"""Typer‑powered CLI wrapper (Phase 3 placeholder).

The CLI currently exposes a single *analyse* command that prints JSON output. In
later phases this will be expanded with rich formatting, multiple sub‑commands
and more options.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print  # type: ignore

from .analyzer import analyse

app = typer.Typer(add_completion=False, help="Analyse English text for sentiment and key phrases.")


@app.command()
def analyse_cmd(
    text: Optional[str] = typer.Argument(None, help="Text to analyse. If omitted, read from stdin or file."),
    input_file: Optional[Path] = typer.Option(
        None, "--input-file", "-i", exists=True, readable=True, help="Path to text file to analyse."
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output results as JSON."),
):
    """Run full analysis and print results."""

    if input_file is not None:
        text_data = input_file.read_text()
    elif text is None:
        import sys

        text_data = sys.stdin.read()
    else:
        text_data = text

    result = analyse(text_data)

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        from pprint import pprint

        pprint(result)


def _main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    _main() 