#!/bin/bash

#  Supported Quantization Types (as of May 2025)
# Common ones include:

# Type	Description
# Q8_0	Highest accuracy, least compression
# Q6_K	Balanced accuracy and size
# Q5_K_M	Good balance for most use cases
# Q4_K_M	Popular choice for fast inference
# Q2_K	Very small, but lowest accuracy


cd llama.cpp &&
apt update &&
apt install cmake build-essential -y &&
apt install -y libcurl4-openssl-dev &&
mkdir build &&
cd build &&
cmake .. &&
cmake --build . --config Release &&
./bin/llama-quantize ../../output/model_trained.gguf ../../output/mistral-lora.Q4_K_M.gguf Q4_K_M