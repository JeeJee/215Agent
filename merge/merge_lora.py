from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from accelerate import dispatch_model, infer_auto_device_map
from huggingface_hub import login
import os
import torch

# Environment
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
login(token=hf_token)

# Paths
base_model_path = "mistralai/Mistral-7B-Instruct-v0.1"
adapter_dir = "output/checkpoint-138"
offload_dir = "offload"
output_dir = "merged_model"

os.makedirs(offload_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu"
)

# Optional: Use offloading if you want to merge on limited RAM
print("Inferring device map...")
device_map = infer_auto_device_map(
    model,
    no_split_module_classes=["MistralDecoderLayer"]
)

print("Dispatching model with offload...")
model = dispatch_model(
    model,
    device_map=device_map,
    offload_dir=offload_dir,
    offload_buffers=True
)

# Load LoRA adapter and merge
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_dir)

print("Merging LoRA weights into base model...")
model = model.merge_and_unload()

# Save model
print("Saving merged model...")
model.save_pretrained(output_dir, safe_serialization=False)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_dir)

print("✅ Merge complete. Output saved to:", output_dir)
