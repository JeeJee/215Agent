# Prompt user for input
$userPrompt = Read-Host "Enter your prompt"

# Create JSON body
$body = @{
    # model = "mistral"
    model = "mistral-lora"
    prompt = $userPrompt
    stream = $false
} | ConvertTo-Json -Depth 3

# Send POST request
$response = Invoke-RestMethod `
    -Uri "http://localhost:11434/api/generate" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"

# Output response
$response
