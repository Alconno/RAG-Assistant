# app/ingest.py
import fitz  # PyMuPDF
import torch
import spacy

# Load a PDF file and extract all text from its pages
def load_pdf(path):
    doc = fitz.open(path)
    text = "\n".join([page.get_text() for page in doc])
    return text

# Load spaCy English model for sentence tokenization
nlp = spacy.load("en_core_web_sm")

# Split raw text into individual sentences using spaCy
def split_text_into_sentences(text, chunkify_sentences: bool = False):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


# Group sentences into meaningful chunks using sentence embeddings.
# Chunks are split based on:
# - cosine similarity drop between consecutive sentence embeddings
# - max token count per chunk (max_chunk_size)
#
# This creates semantically coherent text blocks suitable for embedding or retrieval.
def chunkify(sentences, embeddings, max_chunk_size=192, cosine_similarity_value=0.15):
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for i in range(len(sentences)):
        current_chunk.append(sentences[i])
        current_chunk_size += len(sentences[i].split())
        
        # If next sentence exists, check similarity drop
        if i < len(sentences) - 1:
            emb1 = torch.tensor(embeddings[i])
            emb2 = torch.tensor(embeddings[i+1])

            # Cosine similarity between adjacent sentence embeddings
            sim = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))

            # Split chunk if similarity drops below threshold or max chunk size reached
            if sim < cosine_similarity_value or current_chunk_size >= max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_chunk_size = 0

    # Add leftover sentences as final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


# Create overlapping windows of sentences and generate embeddings for each window.
# Each window contains at least min_sentences to capture more context.
def get_sentence_and_window_embeddings(sentences, embed_fn, min_sentences=15):
    combined_sentences = []
    n = len(sentences)
    k = min(min_sentences, n - 1)

    # Generate overlapping windows of sentences
    for i in range(n - k + 1):
        for j in range(i + k, n):
            if j - i <= k:
                window = " ".join(sentences[i:j])
                combined_sentences.append(window)

    embeddings = embed_fn(combined_sentences)
    return combined_sentences, embeddings


# Compare a query embedding to a list of embeddings, returning the index and score of the closest one.
# Skips None vectors and raises an error if inputs are invalid.
def get_top_cos_sim(query_emb, vecs):
    if query_emb is None:
        raise ValueError("query_emb is None")

    results = []
    for i, v in enumerate(vecs):
        if v is None:
            continue
        sim = torch.cosine_similarity(torch.tensor(query_emb), torch.tensor(v), dim=0)
        results.append([sim.item(), i])

    if not results:
        raise ValueError("No valid vectors to compare.")
    
    # Return [similarity_score, index] of best match
    return max(results, key=lambda x: x[0])
