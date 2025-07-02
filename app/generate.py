from peft import PeftModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re

# Use GPU if available, else CPU (-1)
device = 0 if torch.cuda.is_available() else -1

base_model_path = "Qwen/Qwen3-0.6B"
lora_checkpoint = "./qwen-alpaca-lora/checkpoint-48"

# Load base tokenizer and causal language model with remote code trust enabled
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, device_map="auto")

# Load LoRA (Low-Rank Adaptation) weights on top of the base model
model = PeftModel.from_pretrained(base_model, lora_checkpoint)
model.eval()  # Set model to evaluation mode




def generate_answer(context_chunks, question):
    # Join context chunks with "\nContext:\n" separator if any
    context = "\nContext:\n".join(context_chunks) if len(context_chunks) else ""

    # Compose the prompt using special tokens expected by the model
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Use the following context to answer the question.\n{context}\n\n"
        f"Question: {question}\nAnswer:"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Tokenize prompt and move to model device (GPU/CPU)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response tokens with sampling parameters
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.9,
        top_p=0.95
    )

    # Decode generated tokens including special tokens (skip_special_tokens=False)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("whole output text: ", output_text)

    # Return only the generated text after the prompt, stripped of whitespace
    return output_text[len(prompt):].strip()






def stream_generate_answer(context_chunks, question, temperature=0.9, top_p=0.85):
    # Similar prompt, with an added instruction to limit answer length
    context = "\nContext:\n".join(context_chunks) if len(context_chunks) else ""

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Use the following context to answer the question.\n{context}\n\n"
        f"Question: {question}\nAnswer:\nAnswer in approximately 150 words or less:"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    # Start generation with the prompt input ids
    generated_ids = input_ids.clone()
    past_key_values = None  # Cache for transformer layers to speed up generation

    # Generate tokens one by one, up to max 512 new tokens
    for _ in range(512):
        if past_key_values is None:
            # First token generation
            outputs = model(input_ids=input_ids, use_cache=True)
        else:
            # Subsequent tokens generation using cached past_key_values
            outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)

        logits = outputs.logits  # Shape: [batch=1, seq_len=1, vocab_size]
        past_key_values = outputs.past_key_values  # Update cache

        next_token_logits = logits[0, -1, :]  # Logits for next token prediction

        # Apply temperature and top-p (nucleus) sampling
        probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Filter tokens outside top_p cumulative probability
        filtered_indices = cumulative_probs <= top_p
        filtered_probs = sorted_probs * filtered_indices.float()

        if filtered_probs.sum() == 0:
            # If no tokens remain after filtering, revert to unfiltered probs
            filtered_probs = sorted_probs

        filtered_probs /= filtered_probs.sum()  # Normalize to sum to 1

        # Sample next token id from filtered probability distribution
        next_token = torch.multinomial(filtered_probs, 1)
        next_token_index = next_token.item()
        next_token_id = sorted_indices[next_token_index]

        # Ensure next_token_id is a tensor with batch dimension
        if next_token_id.dim() == 0:
            next_token_id = next_token_id.unsqueeze(0)

        # Reshape to [batch=1, seq_len=1] for concatenation
        next_token_id = next_token_id.unsqueeze(1)

        # Append newly generated token to sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        input_ids = next_token_id  # Next input is just this new token

        # Decode newly generated token (skip special tokens)
        new_token_text = tokenizer.decode(next_token_id.squeeze(1).tolist(), skip_special_tokens=True)
        yield new_token_text  # Yield the token text to the caller for streaming

        # Stop generation if EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break
