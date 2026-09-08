# AutoML-Engine

A dataset-agnostic machine learning system that takes **any CSV file** and automatically handles everything from exploratory data analysis to final predictions — no manual preprocessing required.

## Overview

AutoML-Engine is built to remove the repetitive grind of ML workflows. Drop in a CSV, and the engine:

1. Performs automated **EDA** (missing values, dtype issues, general cleaning)
2. Selects the appropriate **preprocessing pipeline** based on column types
3. Runs **model selection** to find the best-performing algorithm
4. Serves predictions through a clean **Streamlit UI**

The goal is a single, reusable pipeline that works across arbitrary tabular datasets rather than being tuned to one specific problem.

## Core Design Assumptions

- **Target column** is always assumed to be the **last column** in the input CSV.
- **Encoding and scaling** are handled inside `ColumnTransformer` pipeline steps (not applied globally) to prevent data leakage between train/test splits.
- Preprocessing is kept modular so numerical and categorical logic can evolve independently.

### `eda.py`
Handles the first pass over any incoming dataset:
- Detects and handles missing values
- Fixes incorrect/inconsistent dtypes
- General cleaning (e.g. whitespace, duplicate rows, obvious inconsistencies)
- Outputs a cleaned DataFrame ready for the preprocessing stage

### Preprocessing
Numerical and categorical columns are routed through separate pipelines inside a single `ColumnTransformer`, ensuring:
- No leakage of test-set statistics into training
- Consistent, reproducible transformations across train/test/inference

### Model Selection
Automatically evaluates candidate models against the cleaned, preprocessed data and selects the best performer for the given dataset.

### Streamlit UI
Provides a simple interface to:
- Upload any CSV
- View EDA summary and cleaning steps applied
- Get the final model's predictions

## Tech Stack

- **Python**, managed with `uv`
- **pandas** / **NumPy** for data handling
- **scikit-learn** (`ColumnTransformer`, pipelines, models)
- **Streamlit** for the UI

## Getting Started

```bash
# clone the repo
git clone <repo-url>
cd AutoML-Engine

# set up environment with uv
uv sync

# run the Streamlit app
uv run streamlit run app.py
```
