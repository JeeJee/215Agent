#!/bin/bash

set -e  # Exit on any error
set -o pipefail

function step {
  echo ""
  echo "🟡 $1"
}

function success {
  echo "✅ $1"
}

function fail {
  echo "❌ $1"
  exit 1
}

# Step 1: Delete Minikube cluster
step "Deleting Minikube cluster"
minikube delete && success "Minikube deleted"

# Step 2: Start Minikube with Docker driver
step "Starting Minikube with Docker driver"
minikube start --driver=docker && success "Minikube started"

# Step 3: Set Docker env to point to Minikube's Docker
step "Setting Docker environment to Minikube"
eval $(minikube -p minikube docker-env)
success "Docker environment set"

# Step 4: Build Docker image
step "Building Docker image: ollama-local:latest"
docker build -t ollama-local:latest . && success "Docker image built"

# Step 5: Apply deployment
step "Applying 215Agent deployment YAML"
kubectl apply -f ./k8s/215Agent-deployment.yaml && success "Deployment applied"

# Step 6: Apply service
step "Applying 215Agent service YAML"
kubectl apply -f ./k8s/215Agent-service.yaml && success "Service applied"

# Step 7: Wait for pod to be ready
step "Waiting for pod to be ready"
kubectl wait --for=condition=ready pod -l app=215agent --timeout=120s && success "Pod is ready"

# # Step 8: Get service URL
# step "Fetching Minikube service URL"
# minikube service agent-215-service --url || fail "Failed to get service URL"
