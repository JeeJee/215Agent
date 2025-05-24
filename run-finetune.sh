#!/bin/bash

echo "🔧 Building finetune service..."
COMPOSE_BAKE=true docker compose build finetune

echo "▶️ Starting finetune service in detached mode..."
COMPOSE_BAKE=true docker compose up -d finetune
