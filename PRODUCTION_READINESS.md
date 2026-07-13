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

## Phase 1 — Core hardening (planned)

| Issue | Planned fix |
|---|---|
| 58 `except Exception` blocks across `core/`; ~15 in `core/ai_insights_enhanced.py` silently `pass`/`continue` with no logging | Add `logger.warning/error(..., exc_info=True)` at minimum, matching the good pattern already used elsewhere in the same file |
| `utils/jupyter_client.py:61,156` bare `except: pass` | Narrow to specific exceptions + log |
| Duplicate-log-file bug: `setup_logger()` called with default args in 6 places (`core/dimred.py` x3, `core/dimred_evaluator.py`, `core/preprocess.py`, `utils/cloud_executor.py`), each clearing handlers and starting a new timestamped log file | Call `setup_logger()` once at app entry with a stable name; other modules use `get_logger(__name__)` |
| Dead code: `install_packages()`, `ColabServerSetup` template, `RemoteExecutor.execute_automl()`'s local `exec()` fallback (unreferenced from the live app) | Remove, or fold into the Phase 5 remote-execution decision |
| Unpinned dependencies (`>=` only, no lockfile); `setup.py` has placeholder metadata (`author="Your Name"`) and omits half of `requirements.txt`'s deps | Pin versions, fix `setup.py` metadata + `install_requires`, add `.pre-commit-config.yaml`, add `.github/workflows/ci.yml` |

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
