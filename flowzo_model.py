"""
Logistic regression for Flowzo (liquidity risk) — adapted from
Professor Manoel Gadi's template prompt.

Population: Flowzo users who have taken a short-term liquidity advance.

Features (X):
 - time_on_platform_months : discrete, months the customer has used Flowzo
 - total_income_last_12_months : continuous, total income received in the last 12 months
 - total_advances_overdue : continuous, total amount of advances currently overdue

Target (y):
 - repaid_on_time : 1 = repaid the advance on time, 0 = did not repay on time
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

np.random.seed(42)

n = 2000

time_on_platform_months = np.random.randint(1, 61, size=n)  # discrete, 1-60 months
total_income_last_12_months = np.random.normal(loc=18000, scale=5000, size=n).clip(min=0)
total_advances_overdue = np.random.exponential(scale=150, size=n)

# Build a latent risk score (higher = more likely to NOT repay on time)
risk_score = (
    -0.03 * time_on_platform_months
    - 0.00015 * total_income_last_12_months
    + 0.01 * total_advances_overdue
    + np.random.normal(0, 1.0, size=n)
)

prob_not_repaid = 1 / (1 + np.exp(-(risk_score - risk_score.mean()) / risk_score.std()))
not_repaid_on_time = np.random.binomial(1, prob_not_repaid)
repaid_on_time = 1 - not_repaid_on_time  # 1 = good (repaid on time)

df = pd.DataFrame({
    "time_on_platform_months": time_on_platform_months,
    "total_income_last_12_months": total_income_last_12_months,
    "total_advances_overdue": total_advances_overdue,
    "repaid_on_time": repaid_on_time,
})

df.to_excel("flowzo.xlsx", index=False)

X = df[["time_on_platform_months", "total_income_last_12_months", "total_advances_overdue"]]
y = df["repaid_on_time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

print("=== Classification report (TRAIN) ===")
print(classification_report(y_train, model.predict(X_train)))

print("=== Classification report (TEST) ===")
print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "flowzo.joblib")
print("Saved data to flowzo.xlsx and model to flowzo.joblib")
