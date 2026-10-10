# Customer Churn Prediction (MLOps)

An end-to-end machine learning project that predicts which telecom customers
are likely to leave ("churn"). It covers the full workflow: data validation,
feature engineering, hyperparameter tuning, training and evaluation, with
every run tracked in MLflow. A FastAPI prediction service is the next
milestone (see [Roadmap](#roadmap)).

## Why this project

Keeping a customer is cheaper than winning a new one. A retention team can
only contact so many people, so the model's job is to produce a short list of
customers who are most likely to leave.

The two kinds of mistakes are not equally costly:

- **Missing a churner** loses a customer for good.
- **Flagging a loyal customer** only wastes a retention offer.

So this project is tuned to **catch most churners (recall)** while keeping the
flagged list **as accurate as possible (precision)**.

## Highlights

- **One-command pipeline.** Load, validate, preprocess, build features, tune,
  train, evaluate and save, all from `scripts/run_pipeline.py`.
- **Data quality gate.** 23 Great Expectations checks run first. Bad data stops
  the pipeline before any training happens.
- **Recall-first tuning.** Optuna maximises churn precision while guaranteeing
  a minimum churn recall. It also tunes the class weight and the decision
  threshold.
- **No data leakage.** Tuning only ever sees the training split. The test set
  is used once, at the end.
- **Experiment tracking.** Parameters, metrics, dataset and model are logged
  to MLflow for every run.
- **Reproducible.** Fixed seeds throughout, so a rerun gives the same result.
- **Tested.** 50 automated tests cover the pipeline, feature engineering, the
  tuner, the model export, the prediction code and the API.
- **Same features in training and serving.** Feature engineering learns its
  encodings once and saves them, so a single customer is encoded exactly as
  in training.
- **Serving-ready model bundle.** One command exports a small `model/` folder
  (model, feature schema and decision threshold), and a ready-made predictor
  loads it.

## Results

Example run on the held-out test set (1,409 customers, about 27% of whom
churn):

| Metric (churn class)       | Value |
| -------------------------- | ----- |
| Recall                     | 0.837 |
| Precision                  | 0.479 |
| F1 score                   | 0.609 |
| ROC AUC                    | 0.835 |
| Decision threshold (tuned) | 0.469 |

**In plain terms:** the model catches about 84% of the customers who leave
(313 of 374). Of the 654 customers it flags, about 48% really would churn,
roughly 1.8 times better than contacting customers at random.

The minimum recall is a setting (`--min_recall`, default 0.80). Raising it
catches more churners but flags more loyal customers too. Choose it based on
what a retention offer costs compared with losing a customer.

The model was tuned on a validation split and scored on a test set it had
never seen. The two agreed closely, so the result is not an artefact of
tuning.

## How it works

The pipeline runs these stages in order, inside a single MLflow run:

| Stage | What happens | Code |
| ----- | ------------ | ---- |
| 1. Load | Read the raw CSV | `src/data/load_data.py` |
| 2. Validate | 23 Great Expectations checks: required columns, allowed values, numeric ranges, consistency | `src/utils/validate_data.py` |
| 3. Preprocess | Drop the customer ID, fix `TotalCharges` (blank text to number), turn `Churn` into 0/1, fill missing numbers. Saved to `data/processed/` | `src/data/preprocess.py` |
| 4. Build features | Yes/No and two-value columns become 0/1; other categories are one-hot encoded (30 features). The encodings are saved as a schema so serving can repeat them exactly | `src/features/build_features.py` |
| 5. Split | 80% train / 20% test, stratified so both keep the same churn rate | `scripts/run_pipeline.py` |
| 6. Tune | Optuna runs 20 trials on the training data only | `src/models/tune.py` |
| 7. Train | XGBoost is trained with the best settings found | `src/models/train.py` |
| 8. Evaluate | Scored once on the untouched test set | `src/models/evaluate.py` |
| 9. Save | Model and feature list written for serving | `scripts/run_pipeline.py` |

### How tuning works

Each Optuna trial tries a combination of XGBoost settings, a class weight
(how much more a missed churner counts than a missed loyal customer) and a
decision threshold between 0.25 and 0.5.

A trial is judged like this:

1. It must catch at least the minimum share of churners (default 80%).
   Trials that do not are ranked below every trial that does.
2. Among the rest, the highest precision wins.
3. If another trial catches more churners at nearly the same precision
   (within 0.01), that one is preferred.

The search is seeded, so it returns the same answer every time.

### What the exploration found

The analysis lives in `notebooks/EDA.ipynb`. In short:

- About 27% of customers churn, so the classes are imbalanced. Accuracy alone
  would be misleading, which is why recall and precision are the focus.
- Customers with longer tenure or a one/two-year contract churn much less.
- Fiber optic internet and electronic-check payment go with higher churn.
- RandomForest, LightGBM and XGBoost were compared. XGBoost gave similar
  recall to LightGBM and trained about three times faster, so it was chosen.

## Project structure

```
.
├── scripts/
│   ├── run_pipeline.py        # Runs the whole workflow (start here)
│   └── export_model.py        # Exports the trained model as a serving bundle
├── src/
│   ├── data/                  # load_data.py, preprocess.py
│   ├── features/              # build_features.py (fit + transform)
│   ├── models/                # tune.py, train.py, evaluate.py
│   ├── serving/               # inference.py (scores one customer)
│   └── utils/                 # validate_data.py (Great Expectations)
├── model/                     # Serving bundle: model, schema, threshold
├── tests/                     # Pipeline, features, tuner, inference tests
├── notebooks/
│   └── EDA.ipynb              # Exploration and model comparison
├── data/
│   ├── raw/                   # Put the dataset here (not tracked by git)
│   ├── processed/             # Cleaned data written by the pipeline
│   └── external/
├── artifacts/                 # Saved model + feature schema (generated)
├── mlruns/                    # MLflow tracking data (generated)
├── app/                       # FastAPI service (main.py); UI to come (app.py)
├── docker/                    # Dockerfile for the prediction API
├── configs/                   # Reserved for configuration files
├── .github/workflows/         # Planned: CI
└── requirements.txt
```

Folders marked "generated" are created when you run the pipeline. Data,
artifacts and MLflow runs are listed in `.gitignore`, so they are not
committed. The small `model/` bundle is the exception: it is committed so the
service can be built and deployed without retraining.

## Getting started

### Prerequisites

- Python 3.11 (the version this project was developed and tested with)
- Git
- macOS only: the OpenMP library that XGBoost and LightGBM need.
  Install it with `brew install libomp`.

### 1. Download the project

```bash
git clone https://github.com/sadiaparveen-git/customer-churn-prediction_mlops.git
cd customer-churn-prediction_mlops
```

### 2. Create a virtual environment and install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the dataset

The dataset is not stored in the repository. Download the **IBM Telco
Customer Churn** dataset (7,043 customers, 21 columns; it is available on
Kaggle) and save it as:

```
data/raw/Telcom-Customer-Churn.csv
```

The file name matters (note the spelling "Telcom"). To use a different path,
pass `--input path/to/file.csv` in the next step.

### 4. Run the pipeline

```bash
python scripts/run_pipeline.py
```

It takes about a minute. You will see each stage print its progress, and the
test-set classification report at the end.

### 5. Look at the results in MLflow

```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns
```

Then open http://127.0.0.1:5000 in your browser. Each run shows its settings,
metrics, training dataset and saved model. (On Windows PowerShell, first run
`$env:MLFLOW_ALLOW_FILE_STORE="true"`.)

### 6. Run the tests

```bash
pytest tests -v
```

The tests build a small synthetic dataset, so they do not need the real CSV
and they do not touch your own `artifacts/`, `data/processed/` or `mlruns/`.
They take about a minute and a half.

## Configuration

All run-level settings are options on the pipeline script:

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--input` | `data/raw/Telcom-Customer-Churn.csv` | Path to the raw CSV |
| `--target` | `Churn` | Name of the column to predict |
| `--min_recall` | `0.80` | Minimum churn recall tuning must keep |
| `--threshold_min` | `0.25` | Lowest decision threshold to try |
| `--threshold_max` | `0.5` | Highest decision threshold to try |
| `--test_size` | `0.2` | Share of data held out for testing |
| `--experiment` | `Telco Churn - XGBoost` | MLflow experiment name |
| `--mlflow_uri` | `<project>/mlruns` | Where MLflow stores runs |

Example: favour recall more strongly.

```bash
python scripts/run_pipeline.py --min_recall 0.85
```

## What a run produces

| Output | Location |
| ------ | -------- |
| Cleaned data | `data/processed/telco_churn_processed.csv` |
| Trained model | `artifacts/model.joblib` |
| Feature schema: encodings and the exact column order | `artifacts/feature_schema.json` |
| Logged run (settings, metrics, dataset, model) | `mlruns/` |

## Exporting the model for serving

`mlruns/` holds every run and is meant for tracking, not for deployment.
To hand one model to a prediction service, export it:

```bash
python scripts/export_model.py
```

This writes a small bundle to `model/` from the latest finished run (use
`--run_id` to pick another):

| File | Purpose |
| ---- | ------- |
| `model.ubj` | The XGBoost model in its native format (no pickle) |
| `feature_schema.json` | How to build the features: encodings and the exact column order |
| `model_info.json` | Decision threshold, run ID, test metrics, tuned settings and XGBoost version |

The decision threshold lives in `model_info.json`. A service must apply it to
the model's probability, otherwise the recall and precision above will not
hold.

The bundle is about 1.3 MB, so it is committed to the repository. Promoting a
new model is then three steps: run the pipeline, run the export, commit.

## Making a prediction

`ChurnPredictor` loads the bundle and scores one customer from the raw fields,
exactly as they appear in the dataset. Run this from the project root:

```python
from src.serving.inference import ChurnPredictor

predictor = ChurnPredictor()   # loads model/ (or the folder in $MODEL_DIR)

predictor.predict({
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85,
})
# {'churn_score': 0.913, 'likely_to_churn': True,
#  'label': 'Likely to churn', 'threshold': 0.469}
```

- The decision is `churn_score >= threshold`. The score is **not** a
  calibrated probability (the model is trained with a class weight), so show
  the label, not a percentage.
- A missing field or a category the model never saw (for example
  `Contract: "Weekly"`) raises a clear `ValueError` instead of guessing.
- Scoring all 1,409 held-out customers one at a time through this class gives
  the same recall (0.837) and precision (0.479) as the training run.
- This class is what the FastAPI service in `app/` calls, and the planned
  web UI will use it through that API.

## Using your own data

The pipeline structure is generic, but the checks and cleaning are written
for the Telco dataset. To adapt it to other data, review these files first:

- `src/utils/validate_data.py`: the column names, allowed values and ranges.
- `src/data/preprocess.py`: the cleaning steps (for example `TotalCharges`).
- `--target`: the name of your label column (expected to be Yes/No or 0/1).

## Roadmap

- [x] Exploratory analysis and model comparison
- [x] Modular, tested pipeline with data validation
- [x] Hyperparameter tuning with MLflow experiment tracking
- [x] Export of a small, committed serving bundle
- [x] Prediction code shared by training and serving
      (`src/serving/inference.py`)
- [x] FastAPI service that loads the saved model and serves predictions
- [x] Docker image for the service
- [ ] CI with GitHub Actions (tests on every push)

## Troubleshooting

- **`Library not loaded: libomp.dylib` (macOS).** Run `brew install libomp`.
- **MLflow says the file store is deprecated or raises an error.** The pipeline
  already handles this. For `mlflow ui`, set `MLFLOW_ALLOW_FILE_STORE=true` as
  shown above.
- **Port 5000 is already in use (common on macOS, where AirPlay uses it).**
  Add `--port 5001` to the `mlflow ui` command.
- **`File not found: data/raw/...`.** The dataset is missing; see step 3.
- **Pipeline stops with "Data quality check failed".** The data broke one of
  the validation rules. The message lists which checks failed.

## License

Released under the [MIT License](LICENSE).

## Tech stack

Python 3.11, pandas, scikit-learn, XGBoost, Optuna, MLflow, Great Expectations,
pytest. Developed and tested with pandas 3.0, scikit-learn 1.9, XGBoost 3.2,
MLflow 3.16, Great Expectations 1.23 and Optuna 5.0. The requirements file does
not pin versions, so if a future release breaks something, install these
versions.
