#!/bin/bash

# Exit on error
set -e

echo "Setting up MultiLingua-OCR environment..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install -e .

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p results/evaluation

# Download pretrained models and resources (if available)
python -c "
from multilingua_ocr.core.utils import download_resources
download_resources()
"

# Verify CUDA availability
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA is available. Found {torch.cuda.device_count()} device(s).')
    for i in range(torch.cuda.device_count()):
        print(f'Device {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA is not available. Training will proceed on CPU.')
"

echo "Setup completed successfully!"
