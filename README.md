# 🧠 Adobe India Hackathon 2025 – Round 2 "Build and Connect" Submission
This repository contains complete implementations for:

✅ **Round 1A:** PDF Outline Extraction  
✅ **Round 1B:** Persona-Driven Document Intelligence  
Both solutions are **Dockerized**, well-structured, and adhere to the challenge specifications.

----

## Approach

### Heading Extraction ( Challenge 1A)

**Rule-Based:**  
  Uses font size, bold/italic flags, and position to heuristically assign heading levels (see `main.py` and `pdf_processor.py`).

- **Machine Learning:**  
  Trains a classifier on labeled PDF spans to predict heading levels (`train_model.py`, `model_training.py`, `solution.py`).  
  Features include font size, position, style flags, text length, and uppercase ratio.
  
### Persona-Driven Document Intelligence (Challenge 1B)

 **RAG (Retrieval-Augmented Generation)**

- Chunks PDF text and embeds it using [sentence-transformers](https://www.sbert.net/).
- Stores embeddings in a FAISS vector index for fast semantic retrieval.
- Retrieves relevant sections based on a query/task (see `challenge-1b/app/rag/`).

---

## Models & Libraries Used

- **PyMuPDF:** PDF text extraction and layout analysis.
- **scikit-learn:** RandomForestClassifier for heading prediction.
- **sentence-transformers:** Embedding text for semantic search.
- **FAISS:** Efficient vector similarity search for RAG.
- **numpy, pandas:** Data handling and feature engineering.
  
---

## Project Structure

```
Adobe PDF Extracter/
│
├── app/
│   ├── create_training_data.py           # Generate training data from PDFs
│   ├── main.py                           # Rule-based PDF outline extractor
│   ├── make_dataset.py                   # (If present) Dataset creation utility
│   ├── model_training.py                 # (If present) Model training script
│   ├── pdf_processor.py                  # (If present) PDF processing utilities
│   ├── predict_pdf.py                    # ML-based heading prediction
│   ├── solution.py                       # Main solution entrypoint (CLI)
│   ├── __pycache__/                      # Python cache files
│   ├── input/                            # Input PDFs and tasks
│   │   ├── file01.pdf
│   │   └── ... (other PDFs)
│   ├── input_pdfs/                       # (If present) Additional input PDFs
│   ├── output/                           # Output JSONs
│   │   └── file01_output.json
│   ├── data/
│   │   └── training_spans.csv            # Training data CSV
│   ├── model/                            # (If present) Model files
│   ├── models/
│   │   ├── train_model.py                # Model training script
│   │   └── heading_classifier.pkl        # Trained classifier
│   ├── Dokerfile/                        # (Typo? Should be Dockerfile)
│
├── challenge-1b/
│   ├── requirements.txt
│   ├── app/
│   │   ├── input/
│   │   │   ├── task.json                 # Task configuration
│   │   │   └── docs/
│   │   │       ├── South of France - Cities.pdf
│   │   │       ├── South of France - Cuisine.pdf
│   │   │       ├── South of France - History.pdf
│   │   │       └── ... (other PDFs)
│   │   ├── output/
│   │   │   └── output.json
│   │   ├── rag/
│   │   │   ├── chunk_and_embed.py        # Chunking and embedding for RAG
│   │   │   ├── retrieve_and_format.py    # Retrieval and formatting for RAG
│   │   │   ├── vector_store/
│   │   │   │   ├── index.faiss           # FAISS vector index
│   │   │   │   └── metadata.json         # Metadata for chunks
│   │   │   └── ... (other RAG files)
│
├── requirements.txt                      # Top-level dependencies
├── README.md                             # Project documentation
└── ... (other files as needed)
```
------

## Usage

### Generate Training Data

Extracts span-level text features (font size, bold, italic, position, etc.) from all PDFs inside app/input and writes them to app/data/training_spans.csv. You will need to manually label the rows (title, H1–H4) in the label column before training.

python app/create_training_data.py


### Train the Heading Classifier

Trains a RandomForestClassifier on labeled training_spans.csv using engineered features. Automatically performs grid search for best parameters, encodes labels, and saves model (with label encoder) to app/models/heading_classifier.pkl.

python app/models/train_model.py


### Predict Headings/Outline for a PDF

Takes a PDF file as input, extracts spans, runs the trained model on them, and generates a structured outline (title, H1–H4 headings) saved as a JSON file inside app/output/.

python app/predict_pdf.py app/input/file01.pdf


### Rule-Based Extraction 
An optional fallback script that applies rule-based heuristics (font size, indentation, patterns) to extract headings when a model is not available. Useful for debugging or comparison with ML results.

python app/main.py


### RAG Utilities

See the `rag/` folder for advanced document chunking, embedding, and retrieval scripts.
The rag/ folder contains tools for chunking the document semantically, generating embeddings, and querying document sections using vector search. This is useful if you plan to feed the PDF content into a retrieval-augmented generation pipeline (RAG).

-----

## Configuration

- Place input PDFs in the `app/input/` directory.
- Training data is saved to `app/data/training_spans.csv`.
- Model files are stored in `app/models/`.
- Outputs are saved in `app/output/`.

---

## Troubleshooting

- **No features extracted:** Check your input PDF and ensure it contains text.
- **Model errors:** Make sure you have trained the model and the `.pkl` file exists.
- **Feature mismatch:** Regenerate training data and retrain the model if you change features.

---

## Notes

- Place input PDFs in `app/input/` or `challenge-1b/app/input/docs/`.
- Outputs are saved in `app/output/` or `challenge-1b/app/output/`.
- For RAG, ensure vector store files are generated before retrieval.
  
---

## References

- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Label Studio](https://labelstud.io/)
- [RAG (Retrieval-Augmented Generation) Paper](https://arxiv.org/abs/2005.11401)
- [sentence-transformers Documentation](https://www.sbert.net/)

