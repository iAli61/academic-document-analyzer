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

# Copy conda environment file
COPY conda.yaml /app/conda.yaml

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh && \
    ${CONDA_DIR}/bin/conda clean -a -y && \
    ln -s ${CONDA_DIR}/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.js.map' -delete

# Set path to conda
ENV PATH $CONDA_DIR/bin:$PATH

# Create conda environment
RUN conda env create -f conda.yaml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "chunk_caption_index_env", "/bin/bash", "-c"]

# Create directory for tiktoken files
RUN mkdir -p /app/tiktoken_files /app/src/tiktoken_files

# Copy tiktoken downloader script and run it
COPY environment_chunk_caption_download_tiktoken.py /app/download_tiktoken.py
RUN conda run -n chunk_caption_index_env python /app/download_tiktoken.py && \
    cp /app/tiktoken_files/* /app/src/tiktoken_files/

# Copy source files
COPY *.py /app/
COPY src/ /app/src/

# Create component directories
RUN mkdir -p /app/chunk_caption_component \
    /app/chunk_caption_index_component \
    /app/enhanced_doc_analyzer_component

# Default command
CMD ["conda", "run", "--no-capture-output", "-n", "chunk_caption_index_env", "python", "run.py"]