FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git python3 python3-pip python3-venv libaio-dev \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install python-dotenv

# 👇 Specific versions for merging
RUN pip3 install \
    transformers==4.40.1 \
    accelerate==0.29.3 \
    peft==0.11.1 \
    huggingface-hub \
    sentencepiece \
    bitsandbytes \
    scipy \
    scikit-learn \
    tqdm
RUN pip3 install --upgrade peft
RUN pip3 install --upgrade accelerate
RUN pip3 install --upgrade transformers

WORKDIR /app
COPY . .
