#!/bin/bash

set -e  # Exit on any error

accelerate launch finetune.py
