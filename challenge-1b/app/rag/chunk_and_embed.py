import os
import json
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

INPUT_DIR = "challenge-1b/app/input/docs"
OUTPUT_INDEX = "challenge-1b/app/rag/vector_store/index.faiss"
OUTPUT_META = "challenge-1b/app/rag/vector_store/metadata.json"

os.makedirs("app/rag/vector_store", exist_ok=True)

def extract_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("blocks")  
        for block in text:
            clean = block[4].strip()
            if len(clean) > 30:
                blocks.append({
                    "page": page_num + 1,
                    "text": clean
                })
    return blocks

def build_vector_store():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    all_chunks = []
    embeddings = []

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".pdf"):
            continue
        print(f"⏳ Processing: {filename}")
        full_path = os.path.join(INPUT_DIR, filename)
        blocks = extract_text_blocks(full_path)

        for block in blocks:
            chunk_text = block["text"]
            chunk_meta = {
                "document": filename,
                "text": chunk_text,
                "page": block["page"]
            }
            all_chunks.append(chunk_meta)
            embeddings.append(chunk_text)

    print("🔄 Embedding all chunks...")
    vectors = model.encode(embeddings, show_progress_bar=True)
    index = faiss.IndexFlatL2(vectors[0].shape[0])
    index.add(np.array(vectors).astype("float32"))

    print(f"✅ Saved {len(vectors)} chunks to FAISS index.")
    faiss.write_index(index, OUTPUT_INDEX)
    json.dump(all_chunks, open(OUTPUT_META, "w"), indent=2)

if __name__ == "__main__":
    build_vector_store()
