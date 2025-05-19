#!/bin/bash
set -e  # Exit on any error
docker run --rm -it --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark