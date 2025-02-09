Installation Guide
Prerequisites

Python ≥3.8
CUDA ≥11.7 (for GPU support)
16GB RAM minimum
32GB GPU memory recommended

Package Installation
From Source
# Clone repository
git clone https://github.com/multilingua-ocr/framework.git
cd framework

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

Using Docker
# Build and run using docker-compose
docker-compose -f docker/docker-compose.yml up -d

Downloading Resources
from multilingua_ocr.core.utils import download_resources
download_resources()

