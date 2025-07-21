import os
import fitz
import pandas as pd

INPUT_FOLDER = "app/input"
OUTPUT_CSV = "app/data/training_spans.csv"

def get_alignment(x0, page_width):
    if x0 < 0.3 * page_width:
        return "left"
    elif x0 > 0.7 * page_width:
        return "right"
    else:
        return "center"

def auto_label_spans(spans):
    # Step 1: Find title candidate on page 0/1
    title_size = 0
    for span in spans:
        if span["page"] <= 1 and span["bold"] and span["alignment"] == "center":
            if span["size"] > title_size:
                title_size = span["size"]

    # Step 2: map sizes to levels (excluding title)
    font_sizes = sorted({s["size"] for s in spans if s["size"] != title_size}, reverse=True)
    size_to_level = {}
    if font_sizes:
        size_to_level[font_sizes[0]] = "H1"
        if len(font_sizes) > 1:
            size_to_level[font_sizes[1]] = "H2"
        if len(font_sizes) > 2:
            size_to_level[font_sizes[2]] = "H3"
        if len(font_sizes) > 3:
            size_to_level[font_sizes[3]] = "H4"

    # Step 3: Assign labels
    for s in spans:
        if s["size"] == title_size and s["page"] <= 1 and s["bold"] and s["alignment"] == "center":
            s["label"] = "title"
        else:
            s["label"] = size_to_level.get(s["size"], "body")
    return spans

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    spans = []

    for page in doc:
        page_num = page.number
        page_width = page.rect.width
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    bbox = span["bbox"]
                    size = round(span["size"], 2)
                    font = span["font"].lower()

                    spans.append({
                        "text": text,
                        "size": size,
                        "bold": int("bold" in font),
                        "italic": int("italic" in font),
                        "underline": int("underline" in font or "und" in font),
                        "alignment": get_alignment(bbox[0], page_width),
                        "x": round(bbox[0], 2),
                        "page": page_num,
                        "length": len(text),
                        "is_numbered": int(text[:3].replace(".", "").isdigit()) if len(text) >= 3 else 0
                    })

    return auto_label_spans(spans)

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    all_data = []

    for fname in os.listdir(INPUT_FOLDER):
        if fname.endswith(".pdf"):
            print(f"📄 Processing: {fname}")
            spans = process_pdf(os.path.join(INPUT_FOLDER, fname))
            all_data.extend(spans)

    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Training CSV saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()