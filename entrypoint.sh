# #!/bin/sh
# set -e

# MODEL="${OLLAMA_MODEL:-mistral}"
# MAX_RETRIES=3
# COUNT=0

# echo "Pulling model: $MODEL"
# until ollama pull "$MODEL"; do
#     COUNT=$((COUNT + 1))
#     echo "Failed to pull model: $MODEL (attempt $COUNT/$MAX_RETRIES)"
#     [ "$COUNT" -ge "$MAX_RETRIES" ] && exit 1
#     sleep 5
# done

# ollama list | grep -q "$MODEL"

# if [ $? -ne 0 ]; then
#     exit 1
# fi

# echo "Model pulled successfully. Starting Ollama..."
# exec ollama serve

#!/bin/sh

MODEL="${OLLAMA_MODEL:-mistral}"

# Start the Ollama server in the background
ollama serve &

# Wait until the server is accepting connections
echo "Waiting for Ollama to become available..."
until curl -s http://localhost:11434 > /dev/null; do
  sleep 1
done

echo "Ollama is up. Pulling model: ${MODEL}"
ollama pull "${MODEL}"

# Keep the container alive
wait