# InterpretabilityML — Interpreting ML Models for Electric Vehicle Price Prediction

This repository contains a single, end-to-end Jupyter Notebook exploring **model interpretability techniques** on a **regression task**: predicting the **price (EUR)** of an electric vehicle from its technical specifications.

The notebook trains baseline and non-linear models, evaluates performance, then applies **global** and **local** explainability methods to understand *which features drive predictions*.

---

## What’s inside

- **Notebook:** `Interpretability_JuanDavidAlonsoSanabria_Notebook (1).ipynb`
  - Data loading + cleaning
  - Baseline **Linear Regression**
  - **Random Forest Regressor**
  - Model comparison (MAE / MSE / RMSE / R²)
  - Interpretability:
    - **PDP** (Partial Dependence Plots)
    - **ICE** (Individual Conditional Expectation)
    - **ALE** (Accumulated Local Effects)
    - **SHAP** (global summary + local explanations: force / waterfall / decision plots)

---

## Dataset

The notebook expects the CSV file below in the **project root** (same folder as the notebook):

- `ElectricCarData_Clean.csv`

The dataset is referenced in the notebook as coming from Kaggle:

- Kaggle dataset: `geoffnel/evs-one-electric-vehicle-dataset`

> Note: Kaggle downloads typically require a Kaggle account and acceptance of the dataset’s terms.

### Features (high level)

Typical columns include:

- `Brand`, `Model`
- `AccelSec` (0–100 km/h), `TopSpeed_KmH`
- `Range_Km`, `Efficiency_WhKm`
- `FastCharge_KmH`, `RapidCharge`
- `PowerTrain`, `PlugType`, `BodyStyle`, `Segment`, `Seats`
- Target: `PriceEuro`

---

## Quickstart

### 1) Create a Python environment

Recommended: Python **3.9+**.

```bash
python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install numpy pandas scipy matplotlib scikit-learn shap alibi pdpbox
```

For **ALE**, the notebook uses `alepython`. If you have issues with PyPI versions, install from GitHub (as hinted in the notebook):

```bash
pip install git+https://github.com/MaximeJumelle/ALEPython.git@dev#egg=alepython
```

### 3) Add the dataset

Place `ElectricCarData_Clean.csv` in:

```
InterpretabilityML-main/
  ├─ Interpretability_JuanDavidAlonsoSanabria_Notebook (1).ipynb
  ├─ ElectricCarData_Clean.csv   <-- add this
  └─ README.md
```

### 4) Run the notebook

```bash
pip install notebook
jupyter notebook
```

Open `Interpretability_JuanDavidAlonsoSanabria_Notebook (1).ipynb` and run cells top-to-bottom.

---

## Modeling & evaluation

The notebook compares:

- **Linear Regression** (baseline, easier to interpret)
- **Random Forest Regressor** (captures non-linearities and interactions)

Metrics used:

- MAE, MSE, RMSE
- R² on train and test

In the notebook’s run, **Random Forest** outperforms **Linear Regression** on the held-out test split.

---

## Interpretability methods used

### Global explanations

- **PDP (Partial Dependence Plots):** average marginal effect of a feature on predictions.
- **ALE (Accumulated Local Effects):** similar goal to PDP but often more robust under correlated features.

### Local explanations

- **ICE:** per-instance curves showing how changing a feature affects the predicted value.
- **SHAP:**
  - **Summary plot:** global feature importance + directionality
  - **Force plot:** per-instance contributions around the expected value
  - **Waterfall plot:** additive breakdown of a single prediction
  - **Decision plot:** cumulative effect of features along the prediction path

---

## Notes & limitations

- The notebook encodes categorical variables using `LabelEncoder` for convenience. For production-grade modeling, **one-hot encoding** (or target encoding, etc.) is often preferable to avoid imposing artificial ordinal relationships.
- Results may change depending on:
  - Train/test split (the notebook uses `random_state=42`)
  - Random Forest hyperparameters
  - Dataset version/cleaning

---

## How to cite / credit

- Dataset source (as referenced in the notebook): Kaggle `geoffnel/evs-one-electric-vehicle-dataset`
- Some PDP ideas are referenced from a Kaggle notebook: `earije/regression-of-electric-car-data`

---

## License

No license file is included in this repository. If you plan to publish or reuse this code publicly, add a `LICENSE` file (e.g., MIT, Apache-2.0) and ensure the dataset license terms are respected.
