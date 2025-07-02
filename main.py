# streamlit_app.py
import streamlit as st
from app.ingest import split_text_into_sentences, chunkify, get_top_cos_sim, get_sentence_and_window_embeddings
from app.embeddings.embed import embed_text, get_model
from app.generate import generate_answer, stream_generate_answer
from app.weaviate.client import get_weaviate_client
from app.weaviate.utility import split_text_by_token_limit, pdf_to_dataset
from app.weaviate.collections.chunks import create_chunks_collection, batch_insert_chunks, get_top_k_chunks

from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset, concatenate_datasets
import os
import math
import sys

#### Weaviate Settings ####
collection_name = "fineweb_chunks"
data_files =    f"./mypdf.pdf\n" \
                f"C:/Users/Opa/Desktop/my_fineweb_cached/data-00000-of-00099.arrow\n"
#f"C:/Users/Opa/Desktop/my_fineweb_cached/data-00000-of-00099.arrow\n"\
rows_per_iter = 200  # Number of texts processed (split, embed, chunkify, insert) per iteration

print(data_files)

#### App Initialization ####
# Load environment variables and authenticate huggingface API
load_dotenv()
login(os.environ["HUGGINGFACE_APIKEY"])

# Load model and tokenizer once at startup
_model = get_model()
tokenizer = _model.tokenizer

# Setup Weaviate client and ensure collection exists
client = get_weaviate_client()
create_chunks_collection(client, collection_name)

# Streamlit UI setup
st.set_page_config(page_title="RAG UI", layout="centered")
st.title("🧠 Retrieval-Augmented Generation (RAG) UI")

# Sidebar: Dataset ingestion controls
with st.sidebar:
    st.header("Dataset & Processing")
    insert_to_db = st.checkbox("Insert dataset to DB", value=False)
    data_files_text = st.text_area("Arrow/Pdf file paths (\\n-separated):", data_files).splitlines()
    process_btn = st.button("Run Ingestion")


# Data ingestion logic
if process_btn and insert_to_db:
    st.info("Starting ingestion...")

    # Load datasets from provided arrow files and concatenate    
    datasets = []
    for f in data_files_text:
        f = f.strip()
        if not f:
            continue
        if f.endswith(".pdf"):
            # Convert PDF to Dataset
            ds = pdf_to_dataset(f)
            datasets.append(ds)
        else:
            # Load Arrow dataset directly
            ds = Dataset.from_file(f)
            datasets.append(ds)
    #datasets = [Dataset.from_file(f.strip()) for f in data_files_text if f.strip()]
    dataset = concatenate_datasets(datasets)

    n_rows = len(dataset)
    n_chunks = math.ceil(n_rows / rows_per_iter)

    # Progress bar for ingestion process
    progress_bar = st.progress(0, text="Ingesting data...")

    for row_chunk in range(n_chunks):
        start_row_idx = row_chunk * rows_per_iter
        end_row_idx = min(start_row_idx + rows_per_iter, n_rows)

        # Extract texts from current batch slice
        texts_split = [dataset[i]["text"] for i in range(start_row_idx, end_row_idx)]

        # Split each text into smaller chunks respecting token limits
        texts_chunked = []
        for text in texts_split:
            chunks = split_text_by_token_limit(text, tokenizer, custom_limit=50)
            texts_chunked.extend(chunks)

        # Embed the text chunks
        emb_units = embed_text(texts_chunked)

        # Group embeddings & chunks for insertion; filters chunks by similarity threshold
        chunks = chunkify(texts_chunked, emb_units, max_chunk_size=156, cosine_similarity_value=0.2)

        # Batch insert processed chunks into Weaviate
        batch_insert_chunks(client, chunks, collection_name)
        st.success(f"Inserted rows {start_row_idx}–{end_row_idx}")

        # Update progress bar (value between 0.0 and 1.0)
        progress_percent = int((row_chunk + 1) / n_chunks * 100)
        progress_bar.progress(progress_percent / 100.0, text=f"Ingesting data... {progress_percent}%")

    st.success("Ingestion complete ✅")
    progress_bar.empty()  # Clear progress bar after completion


# Main Q&A interface
st.divider()
st.header("🔍 Ask a Question")

# Input for user's query
query = st.text_area("Your question:", "What factors contributed to the reemergence of brucellosis in Bulgaria?")

# Various sliders to tune retrieval/generation parameters
top_chunks = st.slider("Top n Chunks", 1, 10, 2)
st.write("")
result_size_perc = st.slider("Result Sentence Size %", 10, 100, 80)
st.write("")
temperature = st.slider("Temperature", 0.1, 1.0, 0.9)
st.write("")
top_p = st.slider("Top P", 0.1, 1.0, 0.85)
st.write("")
chunk_similarity_threshold = st.slider("Chunk similarity threshold", 0.1, 1.0, 0.85)
st.write("")

submit_btn = st.button("Get Answer")

if submit_btn:
    # Retrieve top-k chunks from Weaviate matching the query and similarity threshold
    top_k_chunks = get_top_k_chunks(client, collection_name, query, k=top_chunks, similarity_threshold=chunk_similarity_threshold)
    chunk_results = []

    # Embed query once for similarity comparisons
    query_emb = embed_text(query)

    for obj in top_k_chunks:
        # Extract raw text from chunk
        result_text = obj.properties.get('data')

        # Split chunk text into sentences for fine-grained selection
        sentences = split_text_into_sentences(result_text)

        # Calculate how many sentences to keep based on slider %
        result_size = math.ceil(len(sentences) * (result_size_perc / 100))

        # Get sentence windows and embeddings (at least result_size sentences)
        all_sentences, all_embeddings = get_sentence_and_window_embeddings(
            sentences, embed_text, min_sentences=result_size)
        
        # Find the sentence with highest cosine similarity to query embedding
        most_similar_sentence_index = get_top_cos_sim(query_emb, all_embeddings)

        # Append the most relevant sentence from chunk to final list
        chunk_results.append(all_sentences[most_similar_sentence_index[1]])

    # Show selected chunk sentences to user for transparency
    st.subheader("📚 Chunks")
    st.write(chunk_results)
    
    # Generate and stream the answer tokens in real-time
    st.subheader("📜 Answer")
    answer_placeholder = st.empty()
    full_answer = ""

    for token in stream_generate_answer(chunk_results, query, temperature, top_p):
        full_answer += token
        answer_placeholder.markdown(full_answer + "▌")  # Display partial answer with blinking cursor

    # Final output without cursor symbol
    answer_placeholder.markdown(full_answer)
