"""Configuration dataclasses used across the *text_analysis* package.

These small objects centralise tweakable constants so future phases can load
and override them from pyproject‑toml or CLI flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class Thresholds:
    """Numeric cut‑offs used in analysis."""

    sentiment: float = 0.4  # abs(VADER compound) to treat word as strongly sentimental


@dataclass(slots=True)
class DomainTerms:
    """Topic‑specific term buckets detected as *key issues*."""

    suggestions: List[str] = field(default_factory=lambda: ["should", "would", "could"])
    business: List[str] = field(default_factory=lambda: ["company", "business", "routes", "overall"])
    workload: List[str] = field(default_factory=lambda: ["busy", "peak", "season", "workload"])
    infrastructure: List[str] = field(
        default_factory=lambda: ["lockers", "setup", "apartment", "complex", "infrastructure"]
    )


@dataclass(slots=True)
class Backends:
    """Backend implementation choices."""

    keyphrase: str = "default"  # "default" | "textrank"
    sentiment: str = "vader"  # "vader" | "transformer"


DEFAULT_THRESHOLDS = Thresholds()
DEFAULT_TERMS = DomainTerms()
DEFAULT_BACKENDS = Backends() 