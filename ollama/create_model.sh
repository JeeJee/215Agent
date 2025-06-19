#!/bin/bash
cd /root/.ollama/models/mistral-lora.Q4_K_M &&
ollama create mistral-lora-q4_k_m --file Modelfile
# ollama run mistral-lora-q4_k_m
