# Customer Churn Intelligence

Customer Churn Intelligence is a practical machine learning application for estimating telecom customer churn risk. It compares three classification models, selects the strongest model using cross-validation, and serves the saved pipeline through FastAPI and a responsive web interface.

## Overview

The application predicts whether a telecom customer may churn based on account, service, contract, and billing information. It handles mixed data types and missing values, applies the same preprocessing during training and inference, and returns a churn probability with a simple risk level.

## Key Features

- Leakage-safe preprocessing for numeric and categorical features.
- Logistic Regression, Random Forest, and XGBoost comparison.
- 5-fold Stratified K-Fold cross-validation using mean and standard deviation for F1 and ROC-AUC.
- Model selection based primarily on mean cross-validated F1.
- Final evaluation on an untouched stratified holdout set.
- FastAPI endpoints for the web application, health checks, and predictions.
- Global feature-importance signals mapped to readable transformed feature names.
- Responsive dark glass frontend built with HTML, CSS, vanilla JavaScript, and Rubik.

## Machine Learning Approach

The workflow is:

`CSV → cleaning → stratified train/test split → 5-fold CV on training data → model selection → final holdout evaluation → pipeline serialization → API inference`

`customerID` is excluded because it is an identifier. Blank `TotalCharges` values are converted to missing values and handled inside the numeric preprocessing pipeline.

The preprocessing and estimator are combined in a single sklearn `Pipeline` for every candidate model:

- Numeric features: median imputation and standardization.
- Categorical features: most-frequent imputation and one-hot encoding with unknown-category support.
- `ColumnTransformer`: applies the correct transformations to each feature group.

The data uses an 80/20 stratified split with `random_state=42`. Model selection is performed using only the training partition with 5-fold `StratifiedKFold`. Mean F1 is the primary selection metric. Mean ROC-AUC is considered only when F1 scores are within the configured practical tie tolerance of `0.005`. The selected pipeline is then fitted on the complete training partition and evaluated once on the untouched holdout set.

## Results

### Cross-validation comparison

| Model | CV F1 mean | CV F1 std | CV ROC-AUC mean | CV ROC-AUC std |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6286 | 0.0231 | 0.8459 | 0.0124 |
| Random Forest | 0.6156 | 0.0214 | 0.8367 | 0.0095 |
| XGBoost | 0.5925 | 0.0258 | 0.8490 | 0.0114 |

Logistic Regression was selected because it achieved the highest mean cross-validated F1-score.

### Final holdout evaluation

After selection, Logistic Regression was fitted on the complete training partition and evaluated on the untouched holdout partition:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8415 |

Cross-validation metrics show average performance and variation across the five training folds. The holdout metrics provide the final one-time evaluation after model selection.

## Explainability

For tree-based models, the application can use `feature_importances_`. For Logistic Regression, it uses absolute transformed coefficients. The preprocessing pipeline supplies the transformed feature names so the strongest signals can be returned with the prediction response.

Feature importance describes how strongly the fitted model relied on a feature. It does not establish that the feature causes churn.

## API

### `GET /`

Serves the web application.

### `GET /health`

Returns the current model loading state:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict`

Accepts customer profile data and returns a prediction, probability, risk level, selected model, and influential signals.

Example request:

```json
{
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "No",
  "dependents": "No",
  "tenure": 12,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "No",
  "streaming_movies": "No",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 79.85,
  "total_charges": 958.2
}
```

Example response from the running application:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.7806,
  "risk_level": "High",
  "model": "Logistic Regression",
  "feature_signals": [
    {"feature": "tenure", "importance": 1.140461, "method": "absolute_coefficient"},
    {"feature": "Contract_Two year", "importance": 0.783232, "method": "absolute_coefficient"},
    {"feature": "InternetService_Fiber optic", "importance": 0.702076, "method": "absolute_coefficient"},
    {"feature": "MonthlyCharges", "importance": 0.672842, "method": "absolute_coefficient"},
    {"feature": "Contract_Month-to-month", "importance": 0.65296, "method": "absolute_coefficient"}
  ]
}
```

The model is loaded once when the FastAPI application starts. Prediction requests only validate the input, apply the saved pipeline, calculate the probability, and construct the response.

## Frontend

The frontend is served directly by FastAPI. It uses a custom dark glass visual design, responsive layout, Rubik typography, native form controls, and lightweight vanilla JavaScript. The interface displays the actual risk level, probability, selected model, progress indicator, interpretation, and feature signals returned by the API.

## Dataset

The project uses the public [Telco Customer Churn dataset](https://github.com/SaeidRostami/Customer_Churn/blob/master/WA_Fn-UseC_-Telco-Customer-Churn.csv), containing customer demographics, services, contracts, payment information, charges, and churn labels. The local copy is stored at `data/telco_churn.csv`.

## Project Structure

```text
customer-churn-intelligence/
├── app.py
├── data/
│   └── telco_churn.csv
├── src/
│   ├── evaluate.py
│   ├── preprocess.py
│   └── train.py
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
├── tests/
│   └── test_api.py
├── requirements.txt
├── sample_request.json
└── .gitignore
```

## Local Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000> after starting the server.

The training command creates `models/churn_model.joblib` and `models/model_metadata.json`, which are excluded from Git and regenerated locally for the application.

## Testing

Run the automated API tests:

```powershell
pytest -q
```

Run a syntax and import compilation check:

```powershell
python -m compileall app.py src tests
```

The test suite verifies model loading, `/health`, a valid `/predict` request, and validation of invalid numeric input.

## Limitations & Future Improvements

This application uses a historical public dataset and does not connect to a live customer system. The default probability bands are not calibrated to a specific business cost matrix, and global feature importance is not a local causal explanation.

Possible future improvements include:

- Probability calibration.
- Business-cost-based threshold tuning.
- Drift monitoring.
- Automated retraining.
- Live CRM or database integration.
- Access controls and deployment authentication.
- More detailed local and model-specific explanations.
