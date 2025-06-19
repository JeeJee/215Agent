from transformers import pipeline

# Load the fine-tuned LoRA model + tokenizer from output dir
model.eval()  # make sure model is in eval mode

# Create text generation pipeline (causal LM)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    temperature=0.7,
    max_new_tokens=50,
)

# Example prompt following your training format
prompt = "Q: What year is it?\nA:"

# Generate response
output = generator(prompt, max_new_tokens=50, do_sample=True)

print("Generated answer:\n", output[0]["generated_text"])
