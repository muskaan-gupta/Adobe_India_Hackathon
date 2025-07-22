import sys
import os
import fitz  # PyMuPDF
import json
import joblib
import time
import pandas as pd

MODEL_PATH = "app/models/heading_classifier.pkl"
OUTPUT_DIR = "app/output"

def extract_features_from_span(span, page_number):
    text = span["text"].strip()
    features = {
        "size": span["size"],
        "bold": int("Bold" in span["font"]),
        "italic": int("Italic" in span["font"]),
        "underline": int(span.get("underline", 0)),
        "x": span["bbox"][0],
        "page": page_number,
        "length": len(text),
        "is_numbered": int(text[:2].strip().split(" ")[0][0].isdigit()) if text else 0
    }
    return features

def extract_spans_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    spans = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        span["page"] = page_num
                        spans.append(span)
    return spans

def main():
    if len(sys.argv) != 2:
        print("❌ Usage: python app/predict_pdf.py <pdf_path>")
        return

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return

    start = time.time()

    spans = extract_spans_from_pdf(pdf_path)
    features = []
    texts = []
    pages = []

    for span in spans:
        feat = extract_features_from_span(span, span["page"])
        features.append(feat)
        texts.append(span["text"].strip())
        pages.append(span["page"])

    df = pd.DataFrame(features)

    # Load trained model
    clf_tuple = joblib.load(MODEL_PATH)
    clf = clf_tuple if not isinstance(clf_tuple, tuple) else clf_tuple[0]
    preds = clf.predict(df)

    # Build output structure
    output = {"title": "", "outline": []}
    title_lines = []

    for text, label, page in zip(texts, preds, pages):
        if label == "title" and page <= 1:
            title_lines.append(text)
        elif label.startswith("H"):
            output["outline"].append({
                "level": label,
                "text": text,
                "page": page + 1  # 1-based indexing
            })

    output["title"] = " ".join(title_lines).strip()

    # Save JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.basename(pdf_path).replace(".pdf", "_output.json")
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Prediction complete. Output saved to: {output_path}")
    print(f"⏱️ Execution Time: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()
