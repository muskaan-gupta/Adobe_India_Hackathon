import pandas as pd
import os
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Paths
INPUT_CSV = "app/data/training_spans.csv"
MODEL_PATH = "app/models/heading_classifier.pkl"

# 1. Load and preprocess data
df = pd.read_csv(INPUT_CSV)
df = df[df["label"].notna() & (df["label"] != "body")]

# Feature matrix and label vector
X = df[["size", "bold", "italic", "underline", "x", "page", "length", "is_numbered"]]
y = df["label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# 3. Define hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

# 4. Apply Grid Search with Cross-Validation
print("\n🔍 Performing Grid Search...")
start = time.time()

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1_weighted",
    verbose=1,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)
duration = time.time() - start

# 5. Use best estimator
clf = grid_search.best_estimator_
print(f"\n✅ Best Hyperparameters: {grid_search.best_params_}")

# 6. Save the model and label encoder
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump((clf, le), MODEL_PATH)
print(f"\n📦 Model saved to {MODEL_PATH}")

# 7. Evaluate performance
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"\n⏱️ Training Time: {duration:.2f} seconds")
print(f"📂 Model File Size: {os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB")
