# docker/Dockerfile

# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# Copy requirements first for better caching
COPY --chown=user:user docker/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user:user . .

# Install the package
RUN pip install -e .

# Set default command
CMD ["bash"]
