# Global Earthquake-Tsunami Risk Assessment

**Dataset:** Global Earthquake-Tsunami Risk Assessment Dataset (Kaggle) — https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset

## Project description
This repository contains a reproducible pipeline to preprocess, explore, and model the Global Earthquake-Tsunami Risk Assessment dataset.
The aim is to build an analytical model to predict tsunami risk / severity or provide geospatial risk insights using available features (magnitude, depth, distance to coast, population, etc.).

## Objectives
- Download and explore the dataset from Kaggle.
- Clean and preprocess the raw data (handle missing values, normalize, encode categorical features).
- Perform exploratory data analysis (visualizations and summary statistics).
- Develop and evaluate one or more predictive models (classification or regression depending on target).
- Write unit tests (pytest) for core data-processing functions.
- Automate tests with GitHub Actions (CI).
- Prepare a short presentation summarizing results (6–10 slides).

## Dataset source
The dataset is available on Kaggle: https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset

> **Note:** The actual CSV files are not included in this repo due to Kaggle licensing and size. Download the dataset manually and place the primary CSV into `data/raw/` as `dataset.csv` before running the pipeline.

## Quickstart (local)
1. Clone repository:
   ```bash
   git clone <your-github-repo-url>
   cd earthquake-tsunami-repo
   ```
2. Create virtual environment and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. Download dataset from Kaggle and put the main CSV at `data/raw/dataset.csv`.
4. Run preprocessing and EDA notebook (Jupyter):
   ```bash
   make env
   make run-notebook
   ```
5. Run tests:
   ```bash
   pytest -q
   ```

## Workflow implemented
- `src/` contains modular code for data loading, preprocessing, EDA, modeling, and evaluation.
- `notebooks/analysis.ipynb` is a runnable EDA & modeling notebook template.
- `tests/` contains Pytest unit tests for core functions.
- `.github/workflows/ci.yml` runs lint and tests on push & pull request.

## How I'll validate final work
After you (the reviewer) validate the final results, I'll prepare a 6–10 slide presentation summarizing objectives, methods, results, and conclusions.

