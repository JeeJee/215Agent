# chat_processor.py

class ChatProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, examples):
        # `examples` is a batch of data items (dict with 'messages' or 'text')
        chat_strings = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in examples["messages"]
        ]
        tokenized = self.tokenizer(
            chat_strings,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    # Optional but recommended, to allow Trainer to decode outputs:
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
