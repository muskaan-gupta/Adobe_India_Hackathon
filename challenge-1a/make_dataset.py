# This script generates a dataset (features and labels) from all PDFs in the input_pdfs directory using the feature extraction and labeling logic from model_training.py.
import os
import numpy as np
from model_training import extract_training_samples_from_pdf

PDF_DIR = 'app/input_pdfs'
X_all = []
y_all = []

for filename in os.listdir(PDF_DIR):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(PDF_DIR, filename)
        X, y = extract_training_samples_from_pdf(pdf_path)
        X_all.extend(X)
        y_all.extend(y)

X_all = np.array(X_all)
y_all = np.array(y_all)

# Save as .npy
np.save('app/model/X_dataset.npy', X_all)
np.save('app/model/y_dataset.npy', y_all)

# Save as .csv for easier use
import pandas as pd
X_df = pd.DataFrame(X_all)
y_df = pd.DataFrame(y_all, columns=["label"])
X_df.to_csv('app/model/X_dataset.csv', index=False)
y_df.to_csv('app/model/y_dataset.csv', index=False)
print(f"Saved dataset: {X_all.shape[0]} samples, {X_all.shape[1] if X_all.shape else 0} features. Also saved as CSV.")
