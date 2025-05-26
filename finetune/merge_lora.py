from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from accelerate import dispatch_model, infer_auto_device_map
import os
import torch
from huggingface_hub import login

# Environment
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
# Log in programmatically
login(token=hf_token)

# Paths
base_model_path = "mistralai/Mistral-7B-v0.1"
adapter_dir = "output/checkpoint-63"  # Path to the LoRA adapter
offload_dir = "offload"
output_dir = "merged_model"

os.makedirs(offload_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    # low_cpu_mem_usage=True
    device_map="cpu"
)

# Infer device map
print("Inferring device map...")
device_map = infer_auto_device_map(
    model,  # ✅ CORRECT: pass the actual model, not the string path
    # max_memory={0: "8GiB", "cpu": "64GiB"},
    no_split_module_classes=["LlamaDecoderLayer"]
)

# Dispatch with offloading support
print("Dispatching model with offload_buffers=True...")
model = dispatch_model(
    model,
    device_map=device_map,
    offload_dir=offload_dir,
    offload_buffers=True
)

# Load LoRA adapter and merge
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_dir)
print("Merging weights...")
model = model.merge_and_unload()

# Save final model and tokenizer
print("Saving merged model...")
model.save_pretrained(output_dir, safe_serialization=False)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_dir)

print("✅ Merge complete. Output saved to:", output_dir)
