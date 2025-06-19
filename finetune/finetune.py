import os
import json
import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from chat_processor import ChatProcessor

# Optional: for avoiding torch._dynamo errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Init accelerator (optional, but harmless here)
accelerator = Accelerator()

# ✅ Load HF token
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set")
login(token=hf_token)
print("✅ Logged in to Hugging Face Hub.")

# ✅ Base model & tokenizer
# model_id = "mistralai/Mistral-7B-v0.1"
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)
print(tokenizer.chat_template)  # Check default chat template;
# === Manually define chat template ===
# tokenizer.chat_template = """{% for message in messages %}
# {% if message['role'] == 'user' %}<|user|>
# {{ message['content'] }}
# {% elif message['role'] == 'assistant' %}<|assistant|>
# {{ message['content'] }}
# {% endif %}
# {% endfor %}"""
tokenizer.pad_token = tokenizer.eos_token  # to avoid warning

# ✅ BitsAndBytes quantization config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ✅ Load 4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    token=hf_token,
)

# ✅ Prepare model for LoRA + 4bit
model = prepare_model_for_kbit_training(model)

# ✅ Setup LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ✅ Apply LoRA wrapper
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ✅ Enable gradient checkpointing
model.gradient_checkpointing_enable()

# ✅ Load and tokenize dataset
data = load_dataset("json", data_files={"train": "data/chat.jsonl"})
for example in data["train"]:
    try:
        _ = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    except Exception as e:
        print(f"❌ Malformed entry: {example['messages']}")
        print(f"Error: {e}")
# def tokenize(example):
#     if isinstance(example["text"], list):
#         example["text"] = [str(text) for text in example["text"]]
#     elif isinstance(example["text"], str):
#         example["text"] = str(example["text"])
#     return tokenizer(example["text"],padding='max_length', truncation=True,max_length=512,return_tensors='pt')


# ✅ Data collator (no MLM for causal LM)
from transformers import DataCollatorWithPadding

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
    return_tensors="pt",
)

# ✅ Training args
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    ddp_find_unused_parameters=False,
    remove_unused_columns=False
)

processor = ChatProcessor(tokenizer)

# ✅ Tokenize the dataset
tokenized_data = data["train"].map(
    processor,
    batched=True,
    remove_columns=data["train"].column_names  # removes 'messages'
)

# ✅ Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=data_collator,
)


tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# ✅ Train
trainer.train()
