# pdf_processor.py
import fitz
import json
import os
import numpy as np
from joblib import load
from model_training import extract_features

class PDFHeadingExtractor:
    def __init__(self):
        self.model = load('app/model/trained_model.joblib')
        self.encoder = load('app/model/label_encoder.joblib')
    
    def predict_heading_level(self, features):
        encoded = self.model.predict([features])
        return self.encoder.inverse_transform(encoded)[0]
    
    def process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        results = {"title": "", "outline": []}
        
        for page in doc:
            text_page = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
            page_height = page.rect.height
            page_width = page.rect.width
            
            for block in text_page["blocks"]:
                if block["type"] != 0:
                    continue
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        if not span["text"].strip():
                            continue
                        
                        features = extract_features(span, page_height, page_width)
                        level = self.predict_heading_level(features)
                        text = span["text"].strip()
                        
                        if level == "title" and page.number == 0 and not results["title"]:
                            results["title"] = text
                        elif level.startswith("H"):
                            results["outline"].append({
                                "level": level,
                                "text": text,
                                "page": page.number + 1
                            })
        
        # Fallback title
        if not results["title"] and len(doc) > 0:
            results["title"] = doc[0].get_text().split('\n')[0].strip()
        
        return results

def process_all_pdfs(input_dir="app/input_pdfs", output_dir="app/output"):
    os.makedirs(output_dir, exist_ok=True)
    extractor = PDFHeadingExtractor()
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            result = extractor.process_pdf(os.path.join(input_dir, filename))
            
            output_file = os.path.join(
                output_dir,
                f"{os.path.splitext(filename)[0]}.json"
            )
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Processed {filename}")

if __name__ == "__main__":
    process_all_pdfs()