import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/llm-embedder")
    #tokenizer = _model.tokenizer
    #print("Context length:", tokenizer.model_max_length)
    return _model

def embed_text(texts):
    model = get_model()
    return model.encode(texts, batch_size=64)