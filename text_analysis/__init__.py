"""Top‑level package for text_analysis refactor.

Exports:
    TextAnalyzer -- main class for analysing text.
    analyse      -- convenience function returning key phrases and sentiment dicts.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("text-analysis")  # type: ignore[arg-type]
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+local"

from .analyzer import TextAnalyzer, analyse  # noqa: E402  # must import after utilities

__all__ = [
    "TextAnalyzer",
    "analyse",
    "__version__",
] 