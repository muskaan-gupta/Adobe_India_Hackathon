import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

# === Paths ===
INDEX_PATH = "challenge-1b/app/rag/vector_store/index.faiss"
META_PATH = "challenge-1b/app/rag/vector_store/metadata.json"
TASK_PATH = "challenge-1b/app/input/task.json"
OUTPUT_PATH = "challenge-1b/app/output/output.json"

# === Load Inputs ===
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)
metadata = json.load(open(META_PATH))
task_data = json.load(open(TASK_PATH))

# === Extract Query Info ===
persona = task_data["persona"]["role"]
job = task_data["job_to_be_done"]["task"]
query = f"{persona} - {job}"

# Get list of allowed PDFs
allowed_filenames = {doc["filename"] for doc in task_data["documents"]}

# === Filter metadata ===
filtered_chunks = [chunk for chunk in metadata if chunk["document"] in allowed_filenames]
filtered_texts = [chunk["text"] for chunk in filtered_chunks]

if not filtered_chunks:
    raise Exception("No valid chunks found for the given documents.")

# === Encode all again in-memory (only allowed docs) ===
print("🔁 Re-embedding chunks for filtered docs...")
chunk_vectors = model.encode(filtered_texts)

# Build temporary FAISS index for filtered chunks
temp_index = faiss.IndexFlatL2(chunk_vectors[0].shape[0])
temp_index.add(np.array(chunk_vectors).astype("float32"))

# === Embed the query ===
query_vec = model.encode([query])[0].astype("float32")

# === Search top-k ===
D, I = temp_index.search(np.array([query_vec]), k=5)
top_chunks = [filtered_chunks[i] for i in I[0]]

# === Format Output ===
output = {
    "metadata": {
        "input_documents": list({chunk["document"] for chunk in top_chunks}),
        "persona": persona,
        "job_to_be_done": job,
        "processing_timestamp": datetime.now().isoformat()
    },
    "extracted_sections": [],
    "subsection_analysis": []
}

for rank, chunk in enumerate(top_chunks, 1):
    output["extracted_sections"].append({
        "document": chunk["document"],
        "section_title": chunk["text"][:50] + "...",
        "importance_rank": rank,
        "page_number": chunk["page"]
    })
    output["subsection_analysis"].append({
        "document": chunk["document"],
        "refined_text": chunk["text"],
        "page_number": chunk["page"]
    })

# === Save output ===
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print("✅ Output written to:", OUTPUT_PATH)
