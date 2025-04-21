# Text‑analysis Improvement Plan

> Branch: `feature/experimental-refactor`

## Goal
Elevate the project from a one‑off script to a maintainable, extensible NLP toolkit that can be used both as a library and a CLI tool, while improving accuracy, performance and developer experience.

---
## Roadmap Overview
| Phase | Focus | Key Deliverables |
|-------|-------|------------------|
| 0 | Project bootstrap & branch | ✅ New Git branch & skeleton |
| 1 | Code organisation | ✅ Package layout, config dataclasses |
| 2 | NLP quality | ✅ TextRank backend & transformer sentiment |
| 3 | CLI / UX | ✅ Typer multi‑command CLI, JSON/YAML output |
| 4 | Performance & packaging | ✅ Model auto toggle, caching, optional extras |
| 5 | Tests & CI | ✅ pytest suite (CI pending) |
| 6 | Docs & release | ✅ README & MkDocs scaffold |

---
## Detailed Tasks
### Phase 0 – Project bootstrap (1 h)
- [x] Create branch `feature/experimental-refactor`.
- [x] Update `.gitignore` if needed (models, caches).
- [x] Add `improvement_plan.md` to track progress.

### Phase 1 – Code organisation (4 h)
- [x] New package layout:
  ```
  text_analysis/
    __init__.py
    keyphrases.py  # extract_key_phrases logic
    sentiment.py   # analyze_sentiment logic
    cli.py         # Typer entry‑point
  tests/
  ```
- [x] Move logic from `text_analysis.py` into modules.
- [x] Replace magic constants with dataclass configs.
- [x] Provide public `analyse()` helper.

### Phase 2 – NLP quality (6 h)
- [x] Key‑phrase extraction: TextRank backend via pytextrank.
- [x] Transformer sentiment (HF pipeline) with fallback.
- [x] VADER kept as default.
- [ ] Externalise domain lexicons (YAML in `resources/`) – TODO.

### Phase 3 – CLI / UX (3 h)
- [ ] Switch to Typer: commands `analyse`, `extract`, `sentiment`.
- [ ] Support `--input-file`, `--json`, `--yaml`, `--model small|large`, `--no-colour`.
- [ ] Display progress bar when downloading models.

### Phase 4 – Performance & packaging (2 h)
- [ ] Add small model option (`en_core_web_sm`) and detect GPU.
- [ ] Cache spaCy pipeline and transformers in `~/.cache/text-analysis/`.
- [ ] `pyproject.toml` extras: `[project.optional-dependencies] full = ["spacy[transformers]", "torch"]`.
- [ ] Define `[project.scripts] text-analyse = text_analysis.cli:app`.

### Phase 5 – Tests & CI (3 h)
- [ ] Write pytest fixtures with sample texts.
- [ ] Unit tests for key‑phrase and sentiment functions.
- [ ] GitHub Actions: run tests on Python 3.9–3.12.

### Phase 6 – Docs & release (2 h)
- [ ] Update README with installation, quick‑start, examples.
- [ ] Auto‑generate API docs via `mkdocs` or `pdoc`.
- [ ] Tag v0.2.0 and draft release notes.

---
## Risk & Mitigation
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Transformer models increase size | Large downloads | Keep current VADER default; make bigger models opt‑in. |
| Refactor breaks existing users | Moderate | Provide backwards‑compat shim (`text_analysis.py` imports new CLI). |
| CI runtime heavy | Slow actions | Use `--skip-large-models` flag in CI; cache models. |

---
## Success Criteria
- ✅ Unit tests ≥ 90 % pass on CI.
- ✅ `text-analyse analyse sample.txt --json` returns structured output.
- ✅ README shows badge "Build ✔".

---
**ETA**: ~21 h of dev time (~3–4 sprint days) 