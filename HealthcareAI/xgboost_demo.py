# xgboost_demo.py
# Demonstrates XGBoost for healthcare: predicting breast cancer (malignant vs. benign)
# Dataset: sklearn's built-in Breast Cancer Wisconsin dataset (569 patients, 30 features)

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE THE DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Load & Explore the Dataset")
print("=" * 60)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 0 = malignant, 1 = benign

print(f"Dataset: Breast Cancer Wisconsin")
print(f"Patients : {X.shape[0]}")
print(f"Features : {X.shape[1]}")
print(f"Target   : 0 = Malignant, 1 = Benign")
print(f"\nClass distribution:")
print(f"  Malignant : {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
print(f"  Benign    : {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
print(f"\nSample features (first 3 rows):")
print(X.iloc[:3, :5].to_string())


# ─────────────────────────────────────────────
# 2. SPLIT THE DATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Train / Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# NOTE: XGBoost handles unscaled data fine, but scaling can help in some cases.
# We skip it here to keep things simple and show XGBoost's raw strength.


# ─────────────────────────────────────────────
# 3. TRAIN THE XGBOOST MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Train XGBoost Classifier")
print("=" * 60)

# Key hyperparameters explained:
#   n_estimators   – number of trees to build (more = stronger, but slower)
#   max_depth      – how deep each tree can grow (controls overfitting)
#   learning_rate  – shrinks each tree's contribution (lower = more conservative)
#   subsample      – fraction of training rows used per tree (reduces overfitting)
#   eval_metric    – what XGBoost optimizes internally

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

model.fit(X_train, y_train)
print("Model trained successfully.")
print(f"  Trees built    : {model.n_estimators}")
print(f"  Max tree depth : {model.max_depth}")
print(f"  Learning rate  : {model.learning_rate}")


# ─────────────────────────────────────────────
# 4. EVALUATE THE MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Evaluate on Test Set")
print("=" * 60)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probability of being Benign

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_prob)

print(f"Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"AUC-ROC  : {auc:.4f}  (1.0 = perfect, 0.5 = random guess)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

# ─────────────────────────────────────────────
# 5. CROSS-VALIDATION (more reliable than a single split)
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 5: 5-Fold Cross-Validation")
print("=" * 60)

# Cross-val trains/tests on 5 different splits and averages the result.
# This gives a more honest picture of real-world performance.
cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

print(f"AUC per fold : {[f'{s:.4f}' for s in cv_scores]}")
print(f"Mean AUC     : {cv_scores.mean():.4f}")
print(f"Std Dev      : {cv_scores.std():.4f}  (lower = more consistent)")


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Top 10 Most Important Features")
print("=" * 60)

# Feature importance tells us which patient measurements matter most
# for the prediction — useful for clinical insight.
importances = pd.Series(model.feature_importances_, index=data.feature_names)
top10 = importances.sort_values(ascending=False).head(10)

print(f"{'Feature':<35} {'Importance':>10}")
print("-" * 47)
for feature, score in top10.items():
    bar = "█" * int(score * 200)
    print(f"{feature:<35} {score:>10.4f}  {bar}")


# ─────────────────────────────────────────────
# 7. MAKE A PREDICTION ON A NEW PATIENT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Predict on a New Patient")
print("=" * 60)

# Grab one real sample from the test set and predict it
sample_index = 0
sample = X_test.iloc[[sample_index]]
actual_label = y_test.iloc[sample_index]

prediction  = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]

label_map = {0: "Malignant", 1: "Benign"}

print(f"Actual diagnosis    : {label_map[actual_label]}")
print(f"Predicted diagnosis : {label_map[prediction]}")
print(f"Confidence          : {max(probability)*100:.2f}%")
print(f"  P(Malignant) = {probability[0]:.4f}")
print(f"  P(Benign)    = {probability[1]:.4f}")
print(f"\nCorrect? {'✓ Yes' if prediction == actual_label else '✗ No'}")

print("\n" + "=" * 60)
print("Done. XGBoost demo complete.")
print("=" * 60)
