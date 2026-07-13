# Production Readiness — Issue Log

Single source of truth for known issues and what's been done about them. Supersedes the older, scattered audit docs now archived under `docs/archive/` (several of which had gone stale or contradicted each other — do not treat them as current).

Status legend: ✅ Fixed · 🟡 In progress · ⬜ Planned

## Phase 0 — Repo hygiene

| Issue | Fix | Status |
|---|---|---|
| 19 overlapping/stale root-level audit `.md` reports | Archived 11 of them to `docs/archive/` (kept README, CONTRIBUTING, GETTING_STARTED, USER_GUIDE, CHANGELOG, PROJECT_STRUCTURE, AGENTS.md, CLOUD_EXECUTION_GUIDE, REMOTE_EXECUTION_GUIDE as current docs) | ✅ |
| Generated run artifacts committed/left in repo root (`automl_exec_*.py`, `automl_runner_*.ipynb`, `automl_results.json`, `automl_dataset.csv` — 4.1MB, `health_check_output.txt`, `pytest_output.txt`) | Deleted; these are regenerable at runtime, not source | ✅ |
| Applied one-off migration scripts left in root (`fix_emojis.py`, `fix_plotly_warnings.py`, `update_streamlit_api.py`) | Deleted — already applied to the codebase, no longer needed | ✅ |
| `.gitignore` didn't actually exclude generated CSVs/artifacts (`# data/*.csv` was commented out) | Added explicit root-anchored ignore rules for `automl_exec_*`, `automl_runner_*`, `automl_results.json`, `automl_dataset.csv`, `health_check_output.txt`, `pytest_output.txt` | ✅ |
| Ad hoc, non-pytest test/demo scripts scattered at repo root (`test_ai.py`, `test_app_features.py`, `test_jupyter_connection.py`, `test_overfitting_fix.py`, `test_fixes.py`) | Moved to `tests/manual/` (print-based smoke scripts, run manually, not part of `pytest tests/`) | ✅ |
| Hardcoded Jupyter token in `test_jupyter_connection.py` (localhost-only, low practical risk, but bad habit for a public repo) | Replaced with `JUPYTER_SERVER_URL`/`JUPYTER_SERVER_TOKEN` env vars, script now fails loudly if unset. Note: the old token string remains in git history from the initial commit — harmless since it's localhost-only and never a real credential, but flagging for awareness | ✅ |
| `health_check.py` was a real reusable diagnostic script sitting loose at root | Moved to `scripts/health_check.py` | ✅ |

## Phase 1 — Core hardening

| Issue | Fix | Status |
|---|---|---|
| 58 `except Exception` blocks across `core/`; 10 in `core/ai_insights_enhanced.py` silently `pass`/`continue` with no logging | Added `logger.debug`/`warning` at each site (debug for expected per-column edge cases, warning for whole-computation failures), preserving the existing defensive fallback behavior | ✅ |
| `utils/jupyter_client.py:61,156` bare `except: pass` | Narrowed to `(ValueError, json.JSONDecodeError)` / `requests.RequestException` + `logger.debug` | ✅ |
| `core/advanced_optimization.py` and `core/tuning.py`: several `except Exception` only `print()`'d gated behind `self.verbose` (default `verbose=True`/`False` varies) — effectively silent when verbose is off | Added unconditional `logger.debug/warning/error` alongside the existing prints | ✅ |
| 6 bare `except:` in `app/ui_dashboard.py` (chart-render fallbacks, clustering label fallbacks) | Narrowed to `Exception as e` + `logger.debug` | ✅ |
| `core/evaluate_cls.py`: `warnings = detector.detect_overfitting(...)` — return value assigned but never read (the actually-used summary comes from a separate `get_user_guidance()` call) | Removed the dead assignment | ✅ |
| Duplicate-log-file bug: `setup_logger()` cleared handlers and opened a new timestamped log file on *every* call, and several classes call it once per instance (`core/dimred.py` x3, `core/dimred_evaluator.py`, `core/preprocess.py`, `utils/cloud_executor.py`) — one run's log ended up scattered across dozens of files | Made `setup_logger()` idempotent: if the named logger already has handlers, return it as-is instead of reconfiguring | ✅ |
| Unpinned dependencies (`>=` only, no lockfile); `setup.py` had placeholder metadata (`author="Your Name"`) and omitted half of `requirements.txt`'s deps (`psutil`, `requests`, `pyngrok`, `python-dotenv`, `groq`, `openai`, `google-generativeai`, `langchain`, `langchain-groq`); `requirements.txt` duplicated dev tools already in `requirements-dev.txt` | `requirements.txt` pinned to exact versions validated in `.venv`; `setup.py` metadata and `install_requires` fixed to match; dev-tool duplication removed from `requirements.txt` | ✅ |
| No formatting/lint consistency (69 of 70 source files would be reformatted by `black`), 318 real flake8 findings (unused imports, f-strings without placeholders, dead variables, a couple of genuine `F811` name-shadowing bugs) | Added `pyproject.toml` (black/isort/mypy/pytest config) and `setup.cfg` (flake8 config); ran `black` + `isort` across `core/`, `app/`, `utils/`, `tests/`, `experiments/`, `scripts/`; removed unused imports/variables (`autoflake`, `ruff --fix`) with one deliberate exception (`utils/cloud_executor.py`'s `import google.colab` is a pure availability probe, kept + `noqa`'d); fixed 2 genuine `F811` redefinitions in `app/ui_dashboard.py`'s PCA tab (dead early imports of names re-imported later). Flake8 is now 0 findings, with a small set of documented `per-file-ignores` in `setup.cfg` for `app/ui_dashboard.py`'s pending Phase 3 debt and for files with long natural-language content strings (LLM prompts/log messages) that shouldn't be hand-wrapped | ✅ |
| No CI, no pre-commit hooks despite `pre-commit` being listed as a dev dependency | Added `.pre-commit-config.yaml` (black, isort, flake8, trailing-whitespace, large-file guard) and `.github/workflows/ci.yml` (lint job: black/isort/flake8 + informational mypy; test job: fast pytest subset) | ✅ |
| `core/dimred.py`, `utils/jupyter_client.py`, `core/advanced_optimization.py` install_packages()/`ColabServerSetup` dead code, `RemoteExecutor.execute_automl()`'s unsandboxed local `exec()` fallback | Deferred to Phase 5 (single decision point for the whole remote-execution feature rather than a partial removal now) | ⬜ |

Known flaky test: `tests/test_models.py::test_mlp_classifier_training` occasionally hits `MemoryError` under full-suite memory pressure (passes reliably in isolation) — a symptom of the suite training real models instead of using lightweight fixtures; addressed in Phase 4.

## Phase 2 — AI engine consolidation (planned)

`core/ai_insights.py` (557 lines) and `core/ai_insights_enhanced.py` (963 lines) are both live and functionally overlapping, not a clean supersession. Plan: merge into one engine.

## Phase 3 — Real UI decomposition (planned)

`app/ui_dashboard.py` is one 8,728-line class (`AutoMLDashboard`, 70 methods), including eight 300–742-line `render_*` methods (~39% of the file: `render_recommendation` 742, `render_explainability` 507, `render_classification_results` 433, `render_advanced_evaluation_sections` 376, `_render_dataset_overview_and_analyzer` 374, `run_automl` 342, `render_report` 314, `run_classification` 301).

A prior refactor pass created `app/tabs/*` (7 files), `app/components/*` (4 files, currently imported nowhere), and `app/state/session_manager.py` (`SessionStateManager`, never referenced — 0 call sites) — but every tab file just delegates back to `self.dashboard.render_X()`. `ui_dashboard.py` has 591 raw `st.session_state` touches (154 direct mutations) that should go through `SessionStateManager` instead.

Plan: move real logic into `app/tabs/*.py` per tab, wire `SessionStateManager` in as each tab migrates, use `app/components/*` where it fits. Verified incrementally, one tab at a time.

## Phase 4 — Test coverage (planned)

120/120 tests pass but the suite takes 7+ minutes (real Optuna trials / full CV loops, not mocked) and has zero coverage on `app/` (UI + tabs), `core/ai_insights.py`, `core/ai_insights_enhanced.py`, `utils/jupyter_client.py`, `utils/cloud_executor.py` — combined >10,000 untested lines.

## Phase 5 — Remote execution feature (planned)

`utils/jupyter_client.py`'s `execute_code_via_file()` has an unsandboxed local `exec()` fallback when remote execution fails. Its only caller path is currently unreferenced from the live app. Decision needed: finish real remote kernel execution properly, or fence the feature as experimental/opt-in and drop the silent local fallback.
