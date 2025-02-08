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

RUN pip3 install -U azure-ai-ml>=1.10
RUN pip3 install -U 'azureml-rag[azure,cognitive_search]'

# Install numpy with specific version first
RUN pip3 install numpy

# Install Nougat from PyPI instead of GitHub
RUN pip3 install nougat-ocr==0.1.17

RUN git clone https://github.com/facebookresearch/nougat.git
WORKDIR /workspace/nougat

RUN python3 setup.py install

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

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; import transformers; import nougat; import pymupdf" || exit 1
