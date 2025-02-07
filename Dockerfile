# Start with the Azure ML PyTorch CUDA base image for GPU support
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-1.13-cuda11.7:latest


# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NOUGAT_CHECKPOINT=/root/.cache/huggingface/hub/models--facebook--nougat-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional Python dependencies
RUN pip3 install --no-cache-dir \
    azure-ai-formrecognizer \
    azure-identity \
    PyMuPDF \
    transformers==4.30.0 \
    Pillow \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    python-dotenv \
    tqdm \
    requests

# Install numpy with specific version first
RUN pip3 install numpy

# Install Nougat from PyPI instead of GitHub
RUN pip3 install nougat-ocr==0.1.17

RUN git clone https://github.com/facebookresearch/nougat.git
WORKDIR /workspace/nougat

RUN python3 setup.py install

# Create necessary directories
RUN mkdir -p /root/.cache/torch/hub/nougat-0.1.0-small

# Download Nougat model files
RUN cd /root/.cache/torch/hub/nougat-0.1.0-small && \
    wget -q https://github.com/facebookresearch/nougat/releases/download/0.1.0-base/config.json && \
    wget -q https://github.com/facebookresearch/nougat/releases/download/0.1.0-base/pytorch_model.bin && \
    wget -q https://github.com/facebookresearch/nougat/releases/download/0.1.0-base/tokenizer.json && \
    wget -q https://github.com/facebookresearch/nougat/releases/download/0.1.0-base/tokenizer_config.json && \
    wget -q https://github.com/facebookresearch/nougat/releases/download/0.1.0-base/special_tokens_map.json



# Fix Nougat transforms configuration for compatibility
# RUN python3 -c "import nougat; import os; \
#     transforms_file = os.path.join(os.path.dirname(nougat.__file__), 'transforms.py'); \
#     with open(transforms_file, 'r') as f: content = f.read(); \
#     content = content.replace('alb.ImageCompression(95, p=0.07),', \
#                             'alb.ImageCompression(quality_lower=95, quality_upper=100, compression_type=\"jpeg\", p=0.07),'); \
#     with open(transforms_file, 'w') as f: f.write(content)"

# Set working directory
WORKDIR /app

# Verify the installation
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    import transformers; print(f'Transformers version: {transformers.__version__}'); \
    import nougat; print(f'Nougat version: {nougat.__version__}'); \
    import pymupdf; print(f'PyMuPDF version: {pymupdf.__version__}')"

# Verify Nougat checkpoint
RUN python3 -c "from nougat.utils.checkpoint import get_checkpoint; print(get_checkpoint())"

# test

# Copy test script
COPY . /app/

RUN ls -la /app/

# Run test script as final step
RUN python3 /app/nb/test.py

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; import transformers; import nougat; import pymupdf" || exit 1
