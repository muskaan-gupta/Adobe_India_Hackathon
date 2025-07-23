
import os
import re
import fitz  # PyMuPDF
import pandas as pd
from collections import Counter

INPUT_FOLDER = "app/input"
OUTPUT_CSV = "app/data/training_spans.csv"

def get_alignment(x0, page_width):
    """Calculates text alignment based on its horizontal position."""
    if x0 < 0.3 * page_width:
        return "left"
    elif x0 > 0.7 * page_width:
        return "right"
    else:
        return "center"

def is_probable_toc_line(text):
    """Checks for Table of Contents-style lines to filter them out."""
    return bool(re.search(r'\.{4,}\s*\d+\s*$', text.strip()))

def is_header_or_footer(bbox, page_height):
    """Checks if a line is in the top or bottom 7% of the page."""
    y0, y1 = bbox[1], bbox[3]
    return y0 < page_height * 0.07 or y1 > page_height * 0.93

def auto_label_spans_advanced(spans):
    """A more intelligent function to label spans as title, H1, H2, or H3."""
    if not spans:
        return []

    font_sizes = [s['size'] for s in spans]
    size_counts = Counter(font_sizes)
    
    if not size_counts:
        return spans
        
    body_size = size_counts.most_common(1)[0][0]
    heading_sizes = sorted([size for size in size_counts if size > body_size], reverse=True)

    size_to_level = {}
    if heading_sizes:
        size_to_level[heading_sizes[0]] = 'H1'
        if len(heading_sizes) > 1:
            size_to_level[heading_sizes[1]] = 'H2'
        if len(heading_sizes) > 2:
            size_to_level[heading_sizes[2]] = 'H3'

    title_found = False
    for s in spans:
        s['label'] = 'body'
        text = s['text'].strip()
        
        match = re.match(r'^(\d+(\.\d+)*)\s', text)
        if match:
            num_dots = match.group(1).count('.')
            if num_dots == 0: s['label'] = 'H1'
            elif num_dots == 1: s['label'] = 'H2'
            else: s['label'] = 'H3'
            continue
            
        if heading_sizes and not title_found and s['page'] == 0 and s['size'] == heading_sizes[0] and s['bold'] and s['alignment'] == 'center':
            s['label'] = 'title'
            title_found = True
            continue

        if s['size'] in size_to_level and s['bold'] and len(text.split()) < 20:
            s['label'] = size_to_level[s['size']]
            
    return spans

def process_pdf(pdf_path):
    """Extracts and processes text lines from a PDF for labeling."""
    doc = fitz.open(pdf_path)
    all_lines = []

    for page_num, page in enumerate(doc):
        page_width = page.rect.width
        page_height = page.rect.height
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH)["blocks"]
        
        for block in blocks:
            if block.get("type") != 0: continue
            for line in block.get("lines", []):
                line_text = " ".join(span["text"].strip() for span in line["spans"])
                if not line_text.strip() or is_probable_toc_line(line_text):
                    continue

                bbox = line["bbox"]
                if is_header_or_footer(bbox, page_height):
                    continue
                
                first_span = line["spans"][0]
                
                # 👇 --- START OF FIX --- 👇
                # Re-added the 'underline' and 'is_numbered' columns
                all_lines.append({
                    "text": line_text,
                    "size": round(first_span["size"], 1),
                    "font": first_span["font"],
                    "bold": int("bold" in first_span["font"].lower()),
                    "italic": int("italic" in first_span["font"].lower()),
                    "underline": 0,  # Underline detection is unreliable, so we add the column with a default value.
                    "alignment": get_alignment(bbox[0], page_width),
                    "x": round(bbox[0], 2),
                    "page": page_num,
                    "length": len(line_text),
                    "is_numbered": int(bool(re.match(r"^\d+(\.\d+)*", line_text)))
                })
                # 👆 --- END OF FIX --- 👆

    return auto_label_spans_advanced(all_lines)

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    all_spans = []
    
    input_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".pdf")]
    print(f"Found {len(input_files)} PDF(s) to process for training data generation.")

    for fname in input_files:
        print(f"Processing {fname}...")
        pdf_path = os.path.join(INPUT_FOLDER, fname)
        spans = process_pdf(pdf_path)
        all_spans.extend(spans)

    if not all_spans:
        print("No data was extracted. Ensure your app/input folder contains PDFs.")
        return
        
    df = pd.DataFrame(all_spans)
    df = df[df['label'] != 'body']
    
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Training CSV saved to {OUTPUT_CSV} with {len(df)} labeled headings.")
    print("Columns generated:", df.columns.tolist())

if __name__ == "__main__":
    main()