from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("output")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

prompt = [{"role": "user", "content": "What year is it?"}]
input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))