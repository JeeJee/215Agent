import os
from huggingface_hub import login
import torch
# sarakie - placa video prea muci
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator()

# Environment
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

if hf_token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set")

# Log in programmatically
login(token=hf_token)

print("Logged in to Hugging Face Hub successfully.")

model_id = "mistralai/Mistral-7B-v0.1"

# Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    token=hf_token
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token  # Avoids warnings

# Load dataset
data = load_dataset("json", data_files={"train": "data/train.jsonl"})
data = data.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Training args
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    eval_strategy="no",
    report_to="none",
    ddp_find_unused_parameters=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    train_dataset=data["train"],
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
)

trainer.train()
