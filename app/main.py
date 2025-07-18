import os
import json
import fitz  # PyMuPDF

INPUT_FOLDER = "app/input"
OUTPUT_FOLDER = "app/output"

# Utility to determine if a line is header/footer
def is_header_footer(y0, y1, page_height):
    margin = 40  # px
    return y0 < margin or y1 > page_height - margin

# Merge multiline titles (simple 2-line case)
def merge_multiline_title(candidates):
    if len(candidates) >= 2 and abs(candidates[0]['size'] - candidates[1]['size']) < 0.5:
        merged = candidates[0]['text'] + " " + candidates[1]['text']
        return merged.strip()
    return candidates[0]['text']

def extract_outline_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_spans = []

    for page_num, page in enumerate(doc, start=0):
        page_height = page.rect.height
        blocks = page.get_text("dict")['blocks']

        for block in blocks:
            if block.get("type") != 0:
                continue  # skip non-text
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text or len(text) < 2:
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    if is_header_footer(y0, y1, page_height):
                        continue
                    all_spans.append({
                        "page": page_num + 1,
                        "text": text,
                        "size": round(span["size"], 1),
                        "font": span["font"],
                        "bold": "Bold" in span["font"],
                        "italic": "Italic" in span["font"],
                        "x": x0,
                        "y": y0
                    })

    # Find unique font sizes and rank
    font_sizes = sorted({s['size'] for s in all_spans}, reverse=True)
    size_to_level = {}

    if font_sizes:
        size_to_level[font_sizes[0]] = "title"
        if len(font_sizes) > 1:
            size_to_level[font_sizes[1]] = "H1"
        if len(font_sizes) > 2:
            size_to_level[font_sizes[2]] = "H2"
        if len(font_sizes) > 3:
            size_to_level[font_sizes[3]] = "H3"

    # Build output
    title_spans = [s for s in all_spans if size_to_level.get(s['size']) == 'title' and s['page'] in (1, 2)]
    title = merge_multiline_title(title_spans) if title_spans else ""

    outline = []
    skip_indexes = set()

    for i, span in enumerate(all_spans):
        if i in skip_indexes:
            continue
        level = size_to_level.get(span["size"])
        if level in ("H1", "H2", "H3"):
            # Check next few spans to see if there's content
            has_content = False
            for j in range(i + 1, min(i + 6, len(all_spans))):
                next_span = all_spans[j]
                if size_to_level.get(next_span['size']) not in ("H1", "H2", "H3") and len(next_span['text']) > 10:
                    has_content = True
                    break
            if has_content:
                outline.append({
                    "level": level,
                    "text": span["text"],
                    "page": span["page"]
                })

    return {
        "title": title,
        "outline": outline
    }

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".pdf"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            result = extract_outline_from_pdf(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✅ Processed: {filename}")

if __name__ == "__main__":
    main()

