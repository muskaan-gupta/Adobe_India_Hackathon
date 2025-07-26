# solution.py
import fitz
import json
import os
import numpy as np
from joblib import load

class HeadingClassifier:
    def __init__(self, model_path, encoder_path):
        self.model = load(model_path)
        self.encoder = load(encoder_path)
    
    def extract_features(self, span, page_height, page_width):
        """Extract features from text span"""
        bbox = span['bbox']
        text = span['text']
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        
        return np.array([
            span['size'],  # Font size
            bbox[1] / page_height,  # Normalized y-position
            (bbox[2] - bbox[0]) / page_width,  # Width ratio
            1 if span['flags'] & 2 else 0,  # Bold flag
            1 if span['flags'] & 8 else 0,  # Italic flag
            len(text),  # Text length
            uppercase_ratio,  # Uppercase ratio
        ]).reshape(1, -1)
    
    def predict(self, features):
        """Predict heading level"""
        encoded = self.model.predict(features)
        return self.encoder.inverse_transform(encoded)[0]

def process_pdf(pdf_path, classifier):
    doc = fitz.open(pdf_path)
    title = ""
    headings = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        page_width = page.rect.width
        
        # Extract text spans
        text_page = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
        
        for block in text_page["blocks"]:
            if block["type"] != 0:  # Skip non-text blocks
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if not span["text"].strip():
                        continue
                    
                    # Extract features and predict
                    features = classifier.extract_features(
                        span, 
                        page_height,
                        page_width
                    )
                    level = classifier.predict(features)
                    
                    # Process prediction
                    text = span["text"].strip()
                    if level == "title" and page_num == 0 and not title:
                        title = text
                    elif level.startswith("H"):
                        headings.append({
                            "level": level,
                            "text": text,
                            "page": page_num + 1
                        })
    
    # Fallback for title
    if not title and doc.page_count > 0:
        first_page = doc[0]
        title = first_page.get_text().split('\n')[0].strip()
    
    return {
        "title": title,
        "outline": headings
    }

def main():
    import sys
    if len(sys.argv) < 5:
        print("Usage: python solution.py <input_pdf> <output_json> <model_path> <encoder_path>")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_json = sys.argv[2]
    model_path = sys.argv[3]
    encoder_path = sys.argv[4]

    classifier = HeadingClassifier(model_path, encoder_path)
    result = process_pdf(input_pdf, classifier)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
    