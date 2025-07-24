# Use ROCm base image for AMD GPU support
FROM rocm/pytorch:latest

WORKDIR /app

# Set environment variables for ROCm
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV PYTORCH_ROCM_ARCH=gfx1031
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV HIP_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    pkg-config \
    libglib2.0-0 \
    libxml2-dev \
    libxslt-dev \
    poppler-utils \
    libopenjp2-7 \
    libjpeg-dev \
    zlib1g-dev \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install ROCm-specific PyTorch first
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.6

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the NiceGUI port
EXPOSE 8000

# Command to run NiceGUI application
CMD ["python3", "main.py"]
