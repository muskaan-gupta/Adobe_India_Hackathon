# 2️⃣ train_model.py (Updated)

import pandas as pd
import os
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

INPUT_CSV = "app/data/training_spans.csv"
MODEL_PATH = "app/models/heading_classifier.pkl"

df = pd.read_csv(INPUT_CSV)
df = df[df["label"].notna() & (df["label"] != "body")]

# --- Start of Changes ---

# Select features, INCLUDING 'alignment'
X = df[["size", "bold", "italic", "underline", "x", "page", "length", "is_numbered", "alignment"]]
y = df["label"]

# Convert 'alignment' into numerical columns (one-hot encoding)
X = pd.get_dummies(X, columns=['alignment'], prefix='align')

# Get the list of columns that the model will be trained on
MODEL_COLUMNS = X.columns.tolist()

# --- End of Changes ---

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

print("\n🔍 Performing Grid Search...")
start = time.time()

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1_weighted",
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
duration = time.time() - start

clf = grid_search.best_estimator_
print(f"\n✅ Best Hyperparameters: {grid_search.best_params_}")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# --- Start of Changes ---
# Save the trained model, label encoder, AND the list of feature columns
joblib.dump((clf, le, MODEL_COLUMNS), MODEL_PATH)
# --- End of Changes ---

print(f"\n📦 Model saved to {MODEL_PATH}")

y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"\n⏱️ Training Time: {duration:.2f} seconds")
print(f"📂 Model File Size: {os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB")