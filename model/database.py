import pandas as pd

# Replace 'data.csv' with your file path
df = pd.read_csv("../dataset/insurance_claims.csv")

# Show first 5 rows
print(df.head())






import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# =========================
# 1️⃣ Load Dataset
# =========================

df = pd.read_csv("../dataset/insurance_claims.csv")

df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

# Select numeric columns
X = df.select_dtypes(include=['int64', 'float64']).copy()
y = df["fraud_reported"]

# Remove target from X
if "fraud_reported" in X.columns:
    X = X.drop(columns=["fraud_reported"])

print("\nClass Distribution:")
print(y.value_counts())

# =========================
# 2️⃣ CLEAN DATA PROPERLY
# =========================

# Replace infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop columns that are ALL NaN
X.dropna(axis=1, how='all', inplace=True)

# Fill remaining NaN with median
X = X.fillna(X.median())

# Final safety check
print("\nAny NaN left?:", X.isnull().sum().sum())

# =========================
# 3️⃣ Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# 4️⃣ Scaling
# =========================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 5️⃣ Models
# =========================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        C=0.5,
        class_weight='balanced'
    ),

    "Random Forest": RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight='balanced',
        random_state=42
    ),

    "SVM": SVC(
        probability=True,
        class_weight='balanced'
    ),

    "Naive Bayes": GaussianNB(),

    "XGBoost": XGBClassifier(
        eval_metric='logloss'
    )
}

results = {}

print("\n===== Model Training & Evaluation =====\n")

# =========================
# 6️⃣ Train & Evaluate
# =========================
# =========================
# 6️⃣ Train & Evaluate (With Custom Threshold)
# =========================

for name, model in models.items():

    # Train model
    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

    # Custom threshold (VERY IMPORTANT)
    threshold = 0.40
    predictions = (probs > threshold).astype(int)

    # Metrics
    acc = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results[name] = f1

    print(f"🔹 {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("--------------------------------------------------")

# =========================
# 7️⃣ Select Best Model
# =========================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"🏆 Best F1 Score: {results[best_model_name]:.4f}")

# =========================
# 8️⃣ Save Model
# =========================

os.makedirs("model", exist_ok=True)

pickle.dump(best_model, open("model/fraud_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("\n✅ Advanced fraud detection model saved successfully!")