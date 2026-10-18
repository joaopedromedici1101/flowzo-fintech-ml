# Flowzo — Liquidity Risk Prediction

Flowzo is a fintech app addressing **Financial and Liquidity Risk** for young people with irregular income (students, gig workers, early-career employees). When Flowzo detects a short-term timing gap between a user's bills and their next paycheck, it fronts a small interest-free cash advance that repays itself once income arrives.

This project is the machine-learning component: a model that predicts **whether a user will repay their advance on time**, so Flowzo's risk team can set advance limits and repayment terms responsibly.

## Problem definition

- **Population:** Flowzo users who have taken out a short-term liquidity advance.
- **ML task:** Binary classification.
- **Target (`repaid_on_time`):** `1` = repaid the advance on time, `0` = did not repay on time.
- **Features:**
  - `time_on_platform_months` (discrete) — months the customer has used Flowzo
  - `total_income_last_12_months` (continuous) — total income received in the last 12 months
  - `total_advances_overdue` (continuous) — total amount of advances currently overdue

## Project structure

```
flowzo-fintech-ml/
├── app.py              # Flask web app: HTML form + JSON API
├── flowzo_model.py      # Generates synthetic data (flowzo.xlsx) and trains the model (flowzo.joblib)
├── flowzo.joblib        # Trained logistic regression model
├── flowzo.xlsx           # Synthetic training data
├── requirements.txt
└── README.md
```

## Setup instructions

1. Clone this repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Regenerate the data and model:
   ```
   python flowzo_model.py
   ```
4. Run the app locally:
   ```
   python app.py
   ```
   Visit `http://127.0.0.1:5000/` to use the form.

### Deploying on PythonAnywhere

1. Upload `app.py` and `flowzo.joblib` into your web app's directory (e.g. `mysite/`).
2. On PythonAnywhere's Flask quickstart, rename/overwrite `flask_app.py` with the contents of `app.py` (or point the WSGI file's import at `app.py`).
3. In a Bash console: `pip install --user -r requirements.txt`.
4. Reload the web app from the **Web** tab.

## Model

Logistic regression trained with scikit-learn on synthetic data (seeded with `numpy.random.seed(42)` for reproducibility) generated to reflect realistic Flowzo usage patterns: more time on the platform, higher income, and lower overdue balances all increase the probability of on-time repayment.

## Web application

### HTML front end — `/`

A form where a user enters `time_on_platform_months`, `total_income_last_12_months`, and `total_advances_overdue`, and receives a live prediction (repaid on time / not repaid on time) along with the probability of on-time repayment.

### API endpoint — `/api/predict`

- **Method:** `POST`
- **Content-Type:** `application/json`
- **Body:**
  ```json
  {
    "time_on_platform_months": 24,
    "total_income_last_12_months": 20000,
    "total_advances_overdue": 50
  }
  ```
- **Response:**
  ```json
  {
    "prediction": 1,
    "prediction_label": "repaid_on_time",
    "probability_repaid_on_time": 0.6334
  }
  ```

#### Example request (curl)

```bash
curl -X POST https://yourusername.pythonanywhere.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"time_on_platform_months": 24, "total_income_last_12_months": 20000, "total_advances_overdue": 50}'
```

#### Example request (Python)

```python
import requests

response = requests.post(
    "https://yourusername.pythonanywhere.com/api/predict",
    json={
        "time_on_platform_months": 24,
        "total_income_last_12_months": 20000,
        "total_advances_overdue": 50,
    },
)
print(response.json())
```
