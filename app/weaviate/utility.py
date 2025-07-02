import hashlib

import pdfplumber
from datasets import Dataset, concatenate_datasets

def pdf_to_dataset(pdf_path):
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append({"text": text})
    return Dataset.from_list(texts)





def uuid_from_string(s):
    return hashlib.md5(s.encode()).hexdigest() 





# Splits a long text into chunks based on a token limit using a tokenizer.
# It ensures that each chunk stays within the model's max token length.

def split_text_by_token_limit(text, tokenizer, max_tokens=None, custom_limit:int = None):
    if max_tokens is None:
        max_tokens = tokenizer.model_max_length if custom_limit == None else custom_limit

    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        token_chunk = tokens[i:i + max_tokens]

        text_chunk = tokenizer.decode(token_chunk, skip_special_tokens=True)

        chunks.append(text_chunk)

    return chunks
