from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "mistralai/Mistral-7B-v0.1"
adapter_dir = "output"

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()

model.save_pretrained("merged_model")
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained("merged_model")
