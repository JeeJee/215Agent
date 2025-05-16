# Step 1: Delete Minikube cluster
Write-Host "🗑️ Deleting Minikube cluster..."
minikube delete

# Step 3: Start Minikube with Docker driver
Write-Host "🚀 Starting Minikube with Docker driver..."
minikube start --driver=docker

# Step 4: Use Minikube Docker daemon for building images
Write-Host "🔧 Switching Docker context to Minikube..."
minikube -p minikube docker-env | Invoke-Expression

# Step 5: Build the latest container image inside Minikube
Write-Host "🐳 Building container image 'ollama-local:latest'..."
docker build -t ollama-local:latest .

# Step 6: Deploy Kubernetes manifests
Write-Host "📦 Deploying Kubernetes manifests..."
kubectl apply -f .\k8s\215Agent-deployment.yaml
kubectl apply -f .\k8s\215Agent-service.yaml