# Start with the Azure ML PyTorch CUDA base image for GPU support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NOUGAT_CHECKPOINT=/root/.cache/huggingface/hub/models--facebook--nougat-base

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get -y install python3-pip git
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# replace CUDA version to your CUDA version.

RUN mkdir workspace
WORKDIR /workspace

RUN pip3 install fastapi uvicorn[standard] fsspec[http]==2023.1.0
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

# Install Python dependencies
RUN pip install --no-cache-dir \
    azure-ai-formrecognizer \
    azure-identity \
    PyMuPDF==1.25.2 \
    transformers==4.30.0 \
    torch==2.5.1 \
    torchvision==0.20.1 \
    Pillow \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    python-dotenv \
    tqdm \
    requests

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

# verify from nougat.utils.checkpoint
RUN python3 -c "from nougat.utils.checkpoint import get_checkpoint; print(get_checkpoint())"

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; import transformers; import nougat; import pymupdf" || exit 1