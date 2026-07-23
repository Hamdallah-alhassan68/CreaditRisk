from flask import Flask, render_template, request
import joblib
import pandas as pd
import sklearn
print(sklearn.__version__)


app = Flask(__name__)


# Load trained model
model = joblib.load("model/credit_risk_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Create a dictionary matching the model's features
    input_data = {
        "person_age": int(request.form["age"]),
        "person_income": float(request.form["income"]),
        "person_emp_length": float(request.form["employment"]),
        "loan_grade": int(request.form["loan_grade"]),
        "loan_amnt": float(request.form["loan_amount"]),
        "loan_int_rate": float(request.form["interest"]),
        "loan_percent_income": float(request.form["loan_percent"]),
        "cb_person_cred_hist_length": int(request.form["credit_history"]),

        # Default all one-hot encoded columns to 0
        "person_home_ownership_OTHER": 0,
        "person_home_ownership_OWN": 0,
        "person_home_ownership_RENT": 0,

        "loan_intent_EDUCATION": 0,
        "loan_intent_HOMEIMPROVEMENT": 0,
        "loan_intent_MEDICAL": 0,
        "loan_intent_PERSONAL": 0,
        "loan_intent_VENTURE": 0,

        "cb_person_default_on_file_Y": 0
    }

    # Home ownership
    home = request.form["home"]

    if home == "OTHER":
        input_data["person_home_ownership_OTHER"] = 1
    elif home == "OWN":
        input_data["person_home_ownership_OWN"] = 1
    elif home == "RENT":
        input_data["person_home_ownership_RENT"] = 1

    # Loan intent
    intent = request.form["intent"]

    if intent == "EDUCATION":
        input_data["loan_intent_EDUCATION"] = 1
    elif intent == "HOMEIMPROVEMENT":
        input_data["loan_intent_HOMEIMPROVEMENT"] = 1
    elif intent == "MEDICAL":
        input_data["loan_intent_MEDICAL"] = 1
    elif intent == "PERSONAL":
        input_data["loan_intent_PERSONAL"] = 1
    elif intent == "VENTURE":
        input_data["loan_intent_VENTURE"] = 1

    # Previous default
    if request.form["default"] == "Y":
        input_data["cb_person_default_on_file_Y"] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = "HIGH RISK" if prediction == 1 else "LOW RISK"

    return render_template(
        "dashboard.html",
        result=result,
        probability=round(probability * 100, 2)
    )
import pandas as pd
import matplotlib.pyplot as plt


feature_importance = pd.read_csv(
    "feature_importance1.csv"
)


plt.figure(figsize=(10,6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "static/feature_importance.png"
)

plt.close()

if __name__ == "__main__":
    app.run(debug=True)