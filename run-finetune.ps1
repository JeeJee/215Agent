# Set COMPOSE_BAKE=true for this session
# $env:COMPOSE_BAKE = "true"

# Build only the finetune service
Write-Host "🔧 Building finetune service..."
docker compose build finetune

# Run the finetune service in detached mode
Write-Host "🚀 Running finetune service in detached mode..."
docker compose up -d finetune

# Show status
Write-Host "`n✅ finetune service is now running in detached mode."
