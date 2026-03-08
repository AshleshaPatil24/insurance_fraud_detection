from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# ==============================
# Load Models
# ==============================

auto_model = pickle.load(open("model/model/fraud_model.pkl", "rb"))
auto_scaler = pickle.load(open("model/model/scaler.pkl", "rb"))

import pickle

health_model = pickle.load(open("model/model/health_model.pkl", "rb"))
health_scaler = pickle.load(open("model/model/health_scaler.pkl", "rb"))

property_model = pickle.load(open("model/model/property_model.pkl", "rb"))
property_scaler = pickle.load(open("model/model/property_scaler.pkl", "rb"))

# ==============================
# LOGIN PAGE
# ==============================

@app.route("/")
def login():
    return render_template("login.html")

# ==============================
# DROPDOWN SELECTION
# ==============================

@app.route("/select", methods=["POST"])
def select():

    insurance_type = request.form["insurance_type"]

    if insurance_type == "auto":
        return redirect(url_for("auto_page"))

    elif insurance_type == "health":
        return redirect(url_for("health_page"))

    elif insurance_type == "property":
        return redirect(url_for("property_page"))

    return redirect(url_for("login"))


# ==============================
# AUTO INSURANCE FRAUD PAGE
# ==============================

@app.route("/auto")
def auto_page():
    # Render the auto insurance page with model accuracy
    return render_template("auto.html", accuracy=85.42)  # replace with your actual model accuracy

# ==============================
# AUTO INSURANCE FRAUD PREDICTION
# ==============================

@app.route("/predict_auto", methods=["POST"])
def predict_auto():

    # Get values from form
    features = [
        float(request.form["age"]),
        float(request.form["months_as_customer"]),
        float(request.form["policy_annual_premium"]),
        float(request.form["total_claim_amount"]),
        float(request.form["incident_hour_of_the_day"]),
        float(request.form["number_of_vehicles_involved"])
    ]

    # Convert to numpy array
    features_array = np.array([features])

    # Scale features
    features_scaled = auto_scaler.transform(features_array)

    # Prediction
    prediction = auto_model.predict(features_scaled)
    probability = auto_model.predict_proba(features_scaled)[0][1]

    # Fraud threshold
    threshold = 0.35

    if probability >= threshold:
        result = "🚨 Fraudulent Auto Claim Detected"
        color = "red"
    else:
        result = "✅ Genuine Auto Claim"
        color = "green"

    # Return result
    return render_template(
        "auto.html",
        result=result,
        probability=round(probability * 100, 2),
        color=color,
        accuracy=85.42  # replace with your actual model accuracy
    )

# ==============================
# HEALTH FRAUD PAGE
# ==============================

@app.route("/health")
def health_page():
    # You can update the accuracy with your best model's F1 or accuracy
    return render_template("health.html", accuracy=75.0)

# ==============================
# HEALTH FRAUD PREDICTION
# ==============================

@app.route("/predict_health", methods=["POST"])
def predict_health():

    # Get values from form
    age = float(request.form["age"])
    months = float(request.form["months_as_customer"])
    premium = float(request.form["policy_annual_premium"])
    claim = float(request.form["total_claim_amount"])

    # Prepare features (same order used during training)
    features = np.array([[age, months, premium, claim]])

    # Scale features
    features_scaled = health_scaler.transform(features)

    # Prediction
    prediction = health_model.predict(features_scaled)
    probability = health_model.predict_proba(features_scaled)[0][1]

    # Fraud threshold
    threshold = 0.35  # you can adjust this

    if probability >= threshold:
        result = "🚨 Fraudulent Claim Detected"
        color = "red"
    else:
        result = "✅ Genuine Claim"
        color = "green"

    # Return result to HTML
    return render_template(
        "health.html",
        result=result,
        probability=round(probability * 100, 2),
        color=color,
        accuracy=75.0  # optional: show model accuracy
    )

# ==============================
# PROPERTY FRAUD PAGE
# ==============================

@app.route("/property")
def property_page():
    return render_template("property.html", accuracy=73.33)

# ==============================
# PROPERTY FRAUD PREDICTION
# ==============================

@app.route("/predict_property", methods=["POST"])
def predict_property():

    # Get values from form
    age = float(request.form["age"])
    months = float(request.form["months_as_customer"])
    premium = float(request.form["policy_annual_premium"])
    claim = float(request.form["total_claim_amount"])

    # Prepare features (same order used during training)
    features = np.array([[age, months, premium, claim]])

    # Scale features
    features_scaled = property_scaler.transform(features)

    # Prediction
    prediction = property_model.predict(features_scaled)
    probability = property_model.predict_proba(features_scaled)[0][1]

    # Fraud threshold
    threshold = 0.35

    if probability >= threshold:
        result = "🚨 Fraudulent Property Claim Detected"
        color = "red"
    else:
        result = "✅ Genuine Property Claim"
        color = "green"

    # Return result
    return render_template(
        "property.html",
        result=result,
        probability=round(probability * 100, 2),
        color=color,
        accuracy=73.33
    )

# ==============================

if __name__ == "__main__":
    app.run(debug=True)
