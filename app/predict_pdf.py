# predict_pdf.py (Updated)

import sys
import os
import fitz  # PyMuPDF
import json
import joblib
import time
import pandas as pd
import re

MODEL_PATH = "app/models/heading_classifier.pkl"
OUTPUT_DIR = "app/output"

def extract_features_from_span(span, page_number, page_width):
    text = span["text"].strip()
    # Simplified alignment logic for consistency with create_training_data.py
    x0 = span["bbox"][0]
    if x0 < 0.3 * page_width:
        alignment = "left"
    elif x0 > 0.7 * page_width:
        alignment = "right"
    else:
        alignment = "center"

    features = {
        "size": span["size"],
        "bold": int("bold" in span["font"].lower()),
        "italic": int("italic" in span["font"].lower()),
        "underline": int("underline" in span["font"].lower()),
        "x": span["bbox"][0],
        "page": page_number,
        "length": len(text),
        "is_numbered": int(bool(re.match(r"^[0-9]+(\\.[0-9]+)*", text))),
        "alignment": alignment
    }
    return features

def extract_spans_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    spans = []
    for page in doc:
        page_num = page.number
        page_width = page.rect.width
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0 or "table" in block.get("text", "").lower():
                continue  # Skip images and tables
            for line in block["lines"]:
                line_text = " ".join(span["text"].strip() for span in line["spans"] if span["text"].strip())
                if re.match(r"^\.{3,}$", line_text) or len(line_text.split()) <= 1:
                    continue  # Skip dotted lines, single-word lines
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        features = extract_features_from_span(span, page_num, page_width)
                        spans.append({"text": text, "page": page_num, **features})
    return spans

def main():
    if len(sys.argv) != 2:
        print("Usage: python app/predict_pdf.py <pdf_path>")
        return

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    start = time.time()
    spans = extract_spans_from_pdf(pdf_path)
    df = pd.DataFrame(spans)

    # --- Start of Changes ---

    # Load the model, encoder, AND the training columns
    clf, le, model_columns = joblib.load(MODEL_PATH)

    # One-hot encode the 'alignment' column in the new data
    df_processed = pd.get_dummies(df, columns=['alignment'], prefix='align')

    # Ensure the new data has the exact same columns as the training data
    # Add any missing columns (e.g., if no 'center' text was found) and fill with 0
    for col in model_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    # Reorder columns to match the model's training order
    df_processed = df_processed[model_columns]

    # Predict using the correctly formatted data
    preds = clf.predict(df_processed)

    # --- End of Changes ---

    labels = le.inverse_transform(preds)

    output = {"title": "", "outline": []}
    title_lines = []
    seen_headings = set()

    merged_headings = ""
    last_label = ""
    last_page = 0

    # Simplified post-processing logic
    for idx, row in df.iterrows():
        label = labels[idx]
        text = row["text"]
        page = row["page"] + 1

        if label == "title":
            title_lines.append(text)
        elif label.startswith("H") and len(text) > 3 and text.lower() not in seen_headings:
            if label == last_label and page == last_page and merged_headings:
                merged_headings += " " + text
            else:
                if merged_headings:
                    output["outline"].append({"level": last_label, "text": merged_headings.strip(), "page": last_page})
                merged_headings = text
                last_label = label
                last_page = page
            seen_headings.add(text.lower())

    if merged_headings:
        output["outline"].append({"level": last_label, "text": merged_headings.strip(), "page": last_page})

    output["title"] = " ".join(title_lines).strip()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(pdf_path).replace(".pdf", "_output.json"))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Prediction complete. Output saved to: {output_path}")
    print(f"Execution Time: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()