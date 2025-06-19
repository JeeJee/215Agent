#!/bin/bash

# set -e  # Exit on any error

accelerate launch finetune.py && accelerate launch merge_lora.py
# pip install -U transformers accelerate datasets peft


# accelerate launch --num_processes 1 --mixed_precision fp16 finetune.py



# git clone https://github.com/ggerganov/llama.cpp
# pip install -r requirements.txt
# python3 convert.py /path/to/your/hf_model --outfile /desired/output/path/model.gguf

# make quantize --> DEPRECATED
# ./quantize ./mistral-lora.gguf ./mistral-lora.Q4_K_M.gguf Q4_K_M

# apt update
# apt install cmake build-essential
# mkdir build
# cd build
# cmake ..
# cmake --build . --config Release
# apt install -y libcurl4-openssl-dev





