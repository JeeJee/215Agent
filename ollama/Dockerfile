FROM ollama/ollama:latest

# Install curl
RUN apt-get update && apt-get install -y curl

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
WORKDIR /

# ENTRYPOINT [ "/bin/sh", "/entrypoint.sh" ]