# import os
# import json
# import fitz  # PyMuPDF

# INPUT_FOLDER = "app/input"
# OUTPUT_FOLDER = "app/output"

# # Utility to determine if a line is header/footer
# def is_header_footer(y0, y1, page_height):
#     margin = 40  # px
#     return y0 < margin or y1 > page_height - margin

# # Merge multiline titles (simple 2-line case)
# def merge_multiline_title(candidates):
#     if len(candidates) >= 2 and abs(candidates[0]['size'] - candidates[1]['size']) < 0.5:
#         merged = candidates[0]['text'] + " " + candidates[1]['text']
#         return merged.strip()
#     return candidates[0]['text']

# def extract_outline_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     all_spans = []

#     for page_num, page in enumerate(doc, start=0):
#         page_height = page.rect.height
#         blocks = page.get_text("dict")['blocks']

#         for block in blocks:
#             if block.get("type") != 0:
#                 continue  # skip non-text
#             for line in block.get("lines", []):
#                 for span in line.get("spans", []):
#                     text = span["text"].strip()
#                     if not text or len(text) < 2:
#                         continue
#                     x0, y0, x1, y1 = span["bbox"]
#                     if is_header_footer(y0, y1, page_height):
#                         continue
#                     all_spans.append({
#                         "page": page_num + 1,
#                         "text": text,
#                         "size": round(span["size"], 1),
#                         "font": span["font"],
#                         "bold": "Bold" in span["font"],
#                         "italic": "Italic" in span["font"],
#                         "x": x0,
#                         "y": y0
#                     })

#     # Find unique font sizes and rank
#     font_sizes = sorted({s['size'] for s in all_spans}, reverse=True)
#     size_to_level = {}

#     if font_sizes:
#         size_to_level[font_sizes[0]] = "title"
#         if len(font_sizes) > 1:
#             size_to_level[font_sizes[1]] = "H1"
#         if len(font_sizes) > 2:
#             size_to_level[font_sizes[2]] = "H2"
#         if len(font_sizes) > 3:
#             size_to_level[font_sizes[3]] = "H3"

#     # Build output
#     title_spans = [s for s in all_spans if size_to_level.get(s['size']) == 'title' and s['page'] in (1, 2)]
#     title = merge_multiline_title(title_spans) if title_spans else ""

#     outline = []
#     skip_indexes = set()

#     for i, span in enumerate(all_spans):
#         if i in skip_indexes:
#             continue
#         level = size_to_level.get(span["size"])
#         if level in ("H1", "H2", "H3"):
#             # Check next few spans to see if there's content
#             has_content = False
#             for j in range(i + 1, min(i + 6, len(all_spans))):
#                 next_span = all_spans[j]
#                 if size_to_level.get(next_span['size']) not in ("H1", "H2", "H3") and len(next_span['text']) > 10:
#                     has_content = True
#                     break
#             if has_content:
#                 outline.append({
#                     "level": level,
#                     "text": span["text"],
#                     "page": span["page"]
#                 })

#     return {
#         "title": title,
#         "outline": outline
#     }

# def main():
#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#     for filename in os.listdir(INPUT_FOLDER):
#         if filename.endswith(".pdf"):
#             input_path = os.path.join(INPUT_FOLDER, filename)
#             output_filename = os.path.splitext(filename)[0] + ".json"
#             output_path = os.path.join(OUTPUT_FOLDER, output_filename)

#             result = extract_outline_from_pdf(input_path)

#             with open(output_path, "w", encoding="utf-8") as f:
#                 json.dump(result, f, indent=2, ensure_ascii=False)
#             print(f"✅ Processed: {filename}")

import os
import json
import re
import fitz  # PyMuPDF

INPUT_FOLDER = "app/input"
OUTPUT_FOLDER = "app/output"

# -- Helper functions --

def is_header_footer(y0, y1, page_height):
    margin = 40
    return y0 < margin or y1 > page_height - margin

def is_centered(span, page_width, tolerance=50):
    text_width_est = len(span['text']) * 6  # rough estimate
    x_center = span['x'] + text_width_est / 2
    return abs(x_center - page_width / 2) < tolerance

def merge_multiline_title(candidates):
    if len(candidates) >= 2 and abs(candidates[0]['size'] - candidates[1]['size']) < 0.5:
        merged = candidates[0]['text'] + " " + candidates[1]['text']
        return merged.strip()
    return candidates[0]['text']

HEADING_PATTERN = re.compile(r"^\d+(\.\d+)*\s+[A-Za-z]")
TOC_LINE_PATTERN = re.compile(r"\.{3,}|\s{3,}|\d{1,2}$")  # Dotted or ends with number

def looks_like_structured_heading(text):
    return bool(HEADING_PATTERN.match(text.strip()))

def is_probable_table_or_toc_line(text):
    """Detect table/TOC-like lines: dotted or ends with page number."""
    text = text.strip()
    return bool(TOC_LINE_PATTERN.search(text.lower()))

# -- Main logic --

def extract_outline_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_spans = []
    title_candidates = []

    for page_num, page in enumerate(doc):
        page_height = page.rect.height
        page_width = page.rect.width
        blocks = page.get_text("dict")['blocks']

        for block in blocks:
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                line_text = " ".join(span["text"].strip() for span in line["spans"] if span["text"].strip())
                if is_probable_table_or_toc_line(line_text):
                    continue  # ❌ Skip table/TOC-like line

                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text or len(text) < 2:
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    if is_header_footer(y0, y1, page_height):
                        continue

                    span_data = {
                        "page": page_num,
                        "text": text,
                        "size": round(span["size"], 1),
                        "font": span["font"],
                        "bold": "Bold" in span["font"],
                        "italic": "Italic" in span["font"],
                        "x": x0,
                        "y": y0,
                        "page_width": page_width
                    }

                    if span_data["bold"] and is_centered(span_data, page_width):
                        title_candidates.append(span_data)

                    all_spans.append(span_data)

    # Determine levels by font size
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

    # Title
    title = merge_multiline_title(title_candidates) if title_candidates else ""

    # Headings
    outline = []
    for i, span in enumerate(all_spans):
        level = size_to_level.get(span["size"])
        text = span["text"]

        is_heading = level in ("H1", "H2", "H3") or looks_like_structured_heading(text)
        if not is_heading:
            continue

        # ❌ Skip heading if no valid content follows
        has_content = False
        for j in range(i + 1, min(i + 6, len(all_spans))):
            next_span = all_spans[j]
            next_text = next_span["text"]
            next_level = size_to_level.get(next_span["size"])

            if next_level in ("H1", "H2", "H3") or looks_like_structured_heading(next_text):
                break
            if not is_probable_table_or_toc_line(next_text) and len(next_text) > 20:
                has_content = True
                break

        if has_content:
            outline.append({
                "level": level if level else "H2",
                "text": text,
                "page": span["page"]
            })

    return {
        "title": title,
        "outline": outline
    }

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

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

