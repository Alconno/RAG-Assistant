# RAG Assistant

**RAG Assistant** is a retrieval-augmented generation (RAG) demo project that enables instruction-following models to answer questions by retrieving relevant knowledge from large datasets.

---

## Key Details

- Data input is expected as Arrow or PDF files (can be both at once).
- Tested and designed to work with FineWeb dataset (e.g. smallest FineWeb version recommended).
- Uses cloud-hosted Weaviate as the vector database (no local Docker setup).
- Frontend is a Streamlit app providing UI for ingestion and querying.
- Environment variables are required for Weaviate API credentials.

---

## Setup and Installation

```bash
git clone https://github.com/Alconno/RAG Assistant.git
cd RAG Assistant
pip install -r requirements.txt
```

Create a `.env` file and fill in your Weaviate API credentials as required.

---

## Usage

1. Download or prepare your dataset in Arrow/PDF file format.  
   For example, use the FineWeb dataset (smallest version recommended).

2. Fine-tune the Qwen-3 to Instruct model (required before running the app):
```bash
python .\app\finetune\Qwen3-0.6B-Instruct\finetune.py
```
3. Run the Streamlit app:  
```bash
streamlit run main.py
```
4. In the opened web UI:  
   - Ingest (process) all or part of your dataset into the vector database (Weaviate).  
   - Once ingestion is complete, enter your questions and receive instruction-following answers enhanced by retrieved knowledge.

---

## Notes

- This is a basic experimental project, so no license, contribution guidelines, or contact info are provided.
- Weaviate vector database is cloud-hosted; ensure your environment variables are properly configured.

---

Enjoy using RAG Assistant for retrieval-augmented instruction generation!
