

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- Configuration ----------------
DATA_PATH = r"C:\Users\Krish Srinivas\Downloads\Mini Project\parkinsons_voice_dataset_augmented.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "parkinson_model.pkl")
RANDOM_STATE = 42

# ---------------- Load Dataset ----------------
print("\nLoading dataset...")
data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)

if "label" not in data.columns:
    raise ValueError("Dataset must contain a 'label' column (0=Healthy, 1=Parkinson’s).")

# ---------------- Features and Target ----------------
y = data["label"]
X = data.drop(["label"], axis=1)

print("Feature count:", X.shape[1])
print("Target distribution:")
print(y.value_counts())

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ---------------- ML Pipeline ----------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE
    ))
])

# ---------------- Train Model ----------------
print("\nTraining model...")
pipeline.fit(X_train, y_train)

# ---------------- Evaluation ----------------
print("\nEvaluating model...")

pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
report = classification_report(y_test, pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# ---------------- Cross Validation ----------------
print("\nPerforming cross-validation...")

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Cross-validation accuracy scores:", cv_scores)
print("Average CV accuracy:", round(cv_scores.mean() * 100, 2), "%")

# ---------------- Feature Importance ----------------
print("\nTop Important Features:")

model = pipeline.named_steps["model"]
importances = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feature_importance.head(10))

# ---------------- Save Model ----------------
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved successfully at:", MODEL_PATH)

print("\nTraining completed successfully.")

