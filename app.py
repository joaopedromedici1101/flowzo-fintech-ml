"""
Flowzo — liquidity risk web application.

Predicts whether a user is likely to repay their short-term Flowzo
liquidity advance on time, based on:
 - time_on_platform_months
 - total_income_last_12_months
 - total_advances_overdue

Serves two things:
 1. An HTML form at "/" for a human to enter values and see a live prediction.
 2. A JSON API at "/api/predict" for other applications to call.

On PythonAnywhere, rename this file to flask_app.py (or point the WSGI
import at this file's module), and keep flowzo.joblib in the same folder.
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowzo.joblib")
model = joblib.load(MODEL_PATH)

PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Flowzo - Liquidity Risk Predictor</title>
</head>
<body style="font-family: sans-serif; max-width: 650px; margin: 40px auto;">
  <h1>Flowzo — Will this advance be repaid on time?</h1>
  <p>
    Flowzo fronts users a short-term, interest-free cash advance when it detects
    a timing gap between their bills and their next paycheck. This tool predicts
    the <strong>probability that a given user will repay their advance on time</strong>,
    based on their history on the platform. Flowzo's risk team uses this prediction
    to decide advance limits and repayment terms, directly addressing the
    Financial and Liquidity Risk this product is built to manage.
  </p>

  <form method="POST" action="/">
    <label>Time on platform (months):</label><br>
    <input type="text" name="time_on_platform_months" value="{time_on_platform_months}"><br><br>

    <label>Total income last 12 months:</label><br>
    <input type="text" name="total_income_last_12_months" value="{total_income_last_12_months}"><br><br>

    <label>Total advances overdue:</label><br>
    <input type="text" name="total_advances_overdue" value="{total_advances_overdue}"><br><br>

    <input type="submit" value="Predict">
  </form>

  {result}

  <hr>
  <p style="color: #666; font-size: 0.9em;">
    API: POST JSON to <code>/api/predict</code> with
    <code>time_on_platform_months</code>, <code>total_income_last_12_months</code>,
    <code>total_advances_overdue</code>. See the README for an example request.
  </p>
</body>
</html>
"""


def run_prediction(time_on_platform_months, total_income_last_12_months, total_advances_overdue):
    X = np.array([[time_on_platform_months, total_income_last_12_months, total_advances_overdue]])
    pred = int(model.predict(X)[0])
    prob_repaid_on_time = float(model.predict_proba(X)[0][1])
    return pred, prob_repaid_on_time


@app.route("/", methods=["GET", "POST"])
def index():
    result_html = ""
    form_values = {
        "time_on_platform_months": "",
        "total_income_last_12_months": "",
        "total_advances_overdue": "",
    }

    if request.method == "POST":
        try:
            time_on_platform_months = float(request.form["time_on_platform_months"])
            total_income_last_12_months = float(request.form["total_income_last_12_months"])
            total_advances_overdue = float(request.form["total_advances_overdue"])

            form_values["time_on_platform_months"] = request.form["time_on_platform_months"]
            form_values["total_income_last_12_months"] = request.form["total_income_last_12_months"]
            form_values["total_advances_overdue"] = request.form["total_advances_overdue"]

            pred, prob_repaid_on_time = run_prediction(
                time_on_platform_months, total_income_last_12_months, total_advances_overdue
            )
            label = "Repaid on time" if pred == 1 else "NOT repaid on time"

            result_html = f"""
            <div style="margin-top:20px; padding:15px; border:1px solid #ccc;">
              <h3>Prediction: {label}</h3>
              <p>Probability of repaying on time: {prob_repaid_on_time:.2%}</p>
            </div>
            """
        except (KeyError, ValueError):
            result_html = "<p style='color:red;'>Please enter valid numbers in all fields.</p>"

    return PAGE_TEMPLATE.format(result=result_html, **form_values)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    required = ["time_on_platform_months", "total_income_last_12_months", "total_advances_overdue"]
    missing = [field for field in required if field not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        time_on_platform_months = float(data["time_on_platform_months"])
        total_income_last_12_months = float(data["total_income_last_12_months"])
        total_advances_overdue = float(data["total_advances_overdue"])
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric"}), 400

    pred, prob_repaid_on_time = run_prediction(
        time_on_platform_months, total_income_last_12_months, total_advances_overdue
    )

    return jsonify({
        "prediction": pred,
        "prediction_label": "repaid_on_time" if pred == 1 else "not_repaid_on_time",
        "probability_repaid_on_time": round(prob_repaid_on_time, 4),
    })


if __name__ == "__main__":
    app.run(debug=True)
