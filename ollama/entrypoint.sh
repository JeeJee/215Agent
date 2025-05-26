#!/bin/sh

set -e

MODEL="${OLLAMA_MODEL:-mistral}"
EMBEDDING="${OLLAMA_EMBEDDING:-nomic-embed-text}"

# Start Ollama server in the background
ollama serve &

# # Wait until the Ollama server responds AND the model list endpoint works
# echo "Waiting for Ollama server to fully start..."
# until curl -s http://localhost:11434/api/tags > /dev/null; do
#   sleep 1
# done

# echo "Ollama is up. Pulling model: ${MODEL}"
# ollama pull "${MODEL}"
# ollama pull "${EMBEDDING}"

# # Verify the models are fully pulled
# echo "Verifying model availability..."
# until curl -s http://localhost:11434/api/tags | grep -q "${MODEL}"; do
#   echo "Waiting for ${MODEL} to be available..."
#   sleep 1
# done

# echo "Models pulled. Starting with model: ${MODEL} and embedding: ${EMBEDDING}"

# echo "FROM mistral\nPARAMETER mistral-lora=mistral-lora.gguf" > /root/.ollama/models/mistral-lora/mistral-lora.modelfile
ollama create mistral-lora --file /root/.ollama/models/mistral-lora/Modelfile
# OPTIONAL: Register the local model
ollama run mistral-lora


# Block forever to keep container running
wait
