# model_training.py
import fitz
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import os

def extract_training_samples_from_pdf(pdf_path):
    """Extract labeled samples from a PDF with manual annotations"""
    doc = fitz.open(pdf_path)
    X = []
    y = []
    
    for page in doc:
        page_height = page.rect.height
        page_width = page.rect.width
        text_page = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
        
        for block in text_page["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if not span["text"].strip():
                        continue
                    
                    text = span["text"].strip()
                    features = extract_features(span, page_height, page_width)
                    
                    # Simple auto-labeling (replace with your actual labels)
                    if span['size'] > 20 and page.number == 0:
                        label = "title"
                    elif span['size'] > 18:
                        label = "H1"
                    elif span['size'] > 16:
                        label = "H2"
                    elif span['size'] > 14:
                        label = "H3"
                    else:
                        label = "body"
                    
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

def extract_features(span, page_height, page_width):
    """Feature extraction helper"""
    bbox = span['bbox']
    text = span['text']
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    
    return [
        span['size'],
        bbox[1] / page_height,
        (bbox[2] - bbox[0]) / page_width,
        1 if span['flags'] & 2 else 0,
        1 if span['flags'] & 8 else 0,
        len(text),
        uppercase_ratio,
    ]

def train_model_from_pdfs(pdf_directory):
    X_all = []
    y_all = []
    
    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            X, y = extract_training_samples_from_pdf(pdf_path)
            X_all.extend(X)
            y_all.extend(y)
    
    # Convert to numpy arrays
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    # Train model with accuracy reporting
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_all)

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=10,  # Lowered to reduce overfitting
        max_depth=3,      # Lowered to reduce overfitting
        random_state=65,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained on {len(X_all)} samples.")
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Save model components
    dump(model, 'app/model/trained_model.joblib')
    dump(encoder, 'app/model/label_encoder.joblib')

if __name__ == "__main__":
    train_model_from_pdfs('app/input_pdfs')