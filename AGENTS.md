# AutoML-Insight Agent Guide

Use this repository from the project root on Python 3.11+.

## What To Know

- Main app entrypoint: [app/main.py](app/main.py) via `streamlit run app/main.py`.
- Experiment CLI: `python experiments/run_experiment.py --config experiments/configs/default.yaml`.
- Keep changes consistent with the existing split between [app/](app), [core/](core), [experiments/](experiments), [tests/](tests), and [utils/](utils).
- Preserve the Windows UTF-8 stdout/stderr workaround in [app/main.py](app/main.py); it avoids `cp1252` failures.
- Run scripts and tests from the repository root; several modules adjust `sys.path` relative to that location.

## Coding Conventions

- Follow [CONTRIBUTING.md](CONTRIBUTING.md): 120-character lines, double quotes, type hints on all public signatures, Google-style docstrings, and one major class per file.
- Prefer small, focused edits and avoid changing unrelated formatting or behavior.
- Keep configuration-driven behavior intact; check [README.md](README.md) and [app/config.yaml](app/config.yaml) before changing defaults.

## Validation

- Preferred test command: `pytest tests/`.
- Targeted test runs are preferred when only one area changes, for example `pytest tests/test_preprocess.py`.
- Style and static checks called out by the repo are `black core/ app/ utils/`, `flake8 core/ app/ utils/`, and `mypy core/ app/ utils/`.
- Use `requirements.txt` and `requirements-dev.txt` for dependency expectations; do not introduce new tooling unless the task requires it.

## Documentation To Link

- [README.md](README.md) for user-facing setup and workflows.
- [CONTRIBUTING.md](CONTRIBUTING.md) for development standards.
- [docs/AI_SETUP.md](docs/AI_SETUP.md) for provider and secret setup.
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) and [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for architecture context.

## Helpful Defaults For Agents

- Prefer `pytest` over ad hoc scripts when you need validation.
- If you touch AI or secret handling, avoid committing credentials and follow the `.env` guidance in the docs.
- If you touch the dashboard, verify the Streamlit app still starts cleanly.