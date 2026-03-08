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

# =========================
# 2️⃣ Select Features
# =========================

features = [
    "age",
    "months_as_customer",
    "policy_annual_premium",
    "total_claim_amount"
]

X = df[features]

# Target
y = df["fraud_reported"].map({"Y":1, "N":0})

print("\nClass Distribution:")
print(y.value_counts())

# =========================
# 3️⃣ Clean Data
# =========================

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(X.median())

print("\nAny NaN left?:", X.isnull().sum().sum())

# =========================
# 4️⃣ Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# =========================
# 5️⃣ Scaling
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 6️⃣ Models
# =========================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
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

print("\n===== Property Fraud Model Training =====\n")

# =========================
# 7️⃣ Train & Evaluate
# =========================

for name, model in models.items():

    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:,1]

    threshold = 0.40
    predictions = (probs > threshold).astype(int)

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
# 8️⃣ Select Best Model
# =========================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n🏆 Best Property Model: {best_model_name}")
print(f"🏆 Best F1 Score: {results[best_model_name]:.4f}")

# =========================
# 9️⃣ Save Model
# =========================

os.makedirs("model", exist_ok=True)

pickle.dump(best_model, open("model/property_model.pkl","wb"))
pickle.dump(scaler, open("model/property_scaler.pkl","wb"))

print("\n✅ Property fraud model saved successfully!")