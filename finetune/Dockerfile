FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Base setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git python3 python3-pip python3-venv libaio-dev \
    && rm -rf /var/lib/apt/lists/*

# 🔧 Fix: make 'python' command available
RUN ln -s /usr/bin/python3 /usr/bin/python

# Python deps
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# HuggingFace + PEFT + LoRA tooling
RUN pip3 install \
    transformers \
    datasets \
    accelerate \
    peft \
    bitsandbytes \
    scipy \
    scikit-learn \
    tqdm \
    sentencepiece

WORKDIR /app
COPY . .

# Optional: if you want to run a shell script later
# RUN chmod +x finetune.sh
# RUN ./finetune.sh

CMD ["python", "finetune.py"]
