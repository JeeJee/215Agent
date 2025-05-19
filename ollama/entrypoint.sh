# #!/bin/sh

# MODEL="${OLLAMA_MODEL:-mistral}"
# EMBEDDING="${OLLAMA_EMBEDDING:-nomic-embed-text}"

# # Start the Ollama server in the background
# ollama serve &

# # Wait until the server is accepting connections
# echo "Waiting for Ollama to become available..."
# until curl -s http://localhost:11434 > /dev/null; do
#   sleep 1
# done

# echo "Ollama is up. Pulling model: ${MODEL}"
# ollama pull "${MODEL}"
# ollama pull "${EMBEDDING}"
# echo "Models pulled. Starting Ollama with model: ${MODEL} and embedding: ${EMBEDDING}"

# # Keep the container alive
# wait

#!/bin/sh

set -e

MODEL="${OLLAMA_MODEL:-mistral}"
EMBEDDING="${OLLAMA_EMBEDDING:-nomic-embed-text}"

# Start Ollama server in the background
ollama serve &

# Wait until the Ollama server responds AND the model list endpoint works
echo "Waiting for Ollama server to fully start..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 1
done

echo "Ollama is up. Pulling model: ${MODEL}"
ollama pull "${MODEL}"
ollama pull "${EMBEDDING}"

# Verify the models are fully pulled
echo "Verifying model availability..."
until curl -s http://localhost:11434/api/tags | grep -q "${MODEL}"; do
  echo "Waiting for ${MODEL} to be available..."
  sleep 1
done

echo "Models pulled. Starting with model: ${MODEL} and embedding: ${EMBEDDING}"

# Block forever to keep container running
wait
