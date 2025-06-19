#!/bin/bash
git clone https://github.com/ggerganov/llama.cpp &&
cd llama.cpp &&
pip install -r requirements.txt &&
python convert_hf_to_gguf.py ../merged_model/ --outfile ../output/model_trained.gguf