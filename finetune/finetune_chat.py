from huggingface_hub import login
import os
from transformers import (
    AutoTokenizer
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ✅ Load HF token
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set")
login(token=hf_token)
print("✅ Logged in to Hugging Face Hub.")

# ✅ Base model & tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)

# ✅ Load and tokenize dataset
chat = load_dataset("json", data_files="data/chat.jsonl", split="train")
print(chat)

tokenized_chat = tokenizer.apply_chat_template(chat["messages"], tokenize=False)
print(tokenizer.decode(tokenized_chat[0])) # ce mortii tei ai facut?

# ✅ Prepare model for LoRA + 4bit
model = prepare_model_for_kbit_training(model_id)