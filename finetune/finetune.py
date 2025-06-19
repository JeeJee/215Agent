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
model_id = "mistralai/Mistral-7B-v0.1"

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)

# === Manually define chat template ===
tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'user' %}<|user|>
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}<|assistant|>
{{ message['content'] }}
{% endif %}
{% endfor %}"""
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
    use_auth_token=hf_token,
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
data = load_dataset("json", data_files={"train": "data/train.jsonl"})

# def tokenize(example):
#     if isinstance(example["text"], list):
#         example["text"] = [str(text) for text in example["text"]]
#     elif isinstance(example["text"], str):
#         example["text"] = str(example["text"])
#     return tokenizer(example["text"],padding='max_length', truncation=True,max_length=512,return_tensors='pt')

def tokenize(batch):
    input_ids = []
    attention_masks = []
    labels = []

    # Support mixed input types: "messages" or "text"
    for i in range(len(batch.get("messages", batch.get("text", [])))):
        if "messages" in batch and isinstance(batch["messages"][i], list):
            prompt = tokenizer.apply_chat_template(
                batch["messages"][i],
                tokenize=False,
                add_generation_prompt=False
            )
            tokenized = tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=512
            )
            input_ids.append(tokenized["input_ids"])
            attention_masks.append(tokenized["attention_mask"])
            labels.append(tokenized["input_ids"])  # label same as input
        elif "text" in batch:
            text = batch["text"][i]
            if isinstance(text, list):
                text = [str(t) for t in text]
            else:
                text = str(text)
            tokenized = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512
            )
            input_ids.append(tokenized["input_ids"])
            attention_masks.append(tokenized["attention_mask"])
            labels.append(tokenized["input_ids"])
        else:
            raise ValueError("Example must have either 'messages' or 'text' field")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }





data = data.map(tokenize, batched=True)

# ✅ Data collator (no MLM for causal LM)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
)

# ✅ Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# ✅ Train
trainer.train()
