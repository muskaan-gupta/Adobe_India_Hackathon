import pandas as pd
import os
import joblib
import time
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

INPUT_CSV = "app/data/training_spans.csv"
MODEL_PATH = "app/models/heading_classifier.pkl"

# 1. Load data
df = pd.read_csv(INPUT_CSV)
df = df[df["label"].notna()]              # remove rows where label is NaN
df = df[df["label"] != "None"]           # remove unlabelled (optional: based on your encoding)

X = df[["size", "bold", "italic", "x", "page", "length", "is_numbered"]]
y = df["label"]

# 2. Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train model
start_time = time.time()
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
training_time = time.time() - start_time

# 4. Evaluate on test set
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 5. Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")

# 6. Report performance
model_size_bytes = os.path.getsize(MODEL_PATH)
model_size_mb = model_size_bytes / (1024 * 1024)
print(f"\n⏱️ Training Time: {training_time:.2f} seconds")
print(f"📦 Model Size: {model_size_mb:.2f} MB")