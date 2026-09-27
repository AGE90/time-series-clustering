# Time Series Clustering

Reference repo for time series clustering techniques

## Conventions

- Python package: `src/tsclustering/` (data, features, models, visualization, utils).
- Environment: uv. Run everything with `uv run <cmd>`; add deps with `uv add <pkg>` (or `uv add --group <dev|test|notebook|data-science|viz> <pkg>`). Never use pip directly.
- Paths: never hardcode. Use the helpers in `tsclustering.utils.paths` (`data_raw_dir("file.csv")`, `data_processed_dir(...)`, `models_dir(...)`, `reports_figures_dir(...)`, ...).
- Data flow: `data/raw` is immutable input -> `data/interim` -> `data/processed` (model-ready). Third-party data goes in `data/external`. Data files are not committed to git.
- Notebooks in `notebooks/` are for exploration only; move reusable code into `src/`.
- Trained models go in `models/`, figures in `reports/figures/`.
- Secrets go in `.env` (git-ignored), loaded via `tsclustering.credentials`.
- Each new technique = one module/`MODELS` entry + a unit test + a numbered notebook; update the README technique table and ROADMAP.md.
- Docstrings: numpy style. Type hints on public functions.

## Commands

- `make install`: install all dependency groups
- `make data` / `make features` / `make train` / `make predict`: pipeline steps (`make pipeline` runs the first three). `DATASET` is set in `data/make_dataset.py`; clustering techniques are entries in `MODELS` in `models/train_model.py`
- `make check`: ruff format + ruff check + mypy
- `make test`: pytest with coverage (tests live in `tests/unit` and `tests/e2e`)
- `make help`: list every target
