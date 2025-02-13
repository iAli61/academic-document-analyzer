# Use Python image as builder to download tiktoken files
FROM python:3.9 as builder

# Install tiktoken
RUN pip install tiktoken

# Download tiktoken files in the builder
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Get the tiktoken directory with downloaded files
RUN cp -r /root/.cache/huggingface/hub /tiktoken_cache

# Main image
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# Copy conda environment file
COPY conda.yaml /tmp/conda.yaml

# Create conda environment
RUN conda env create -f /tmp/conda.yaml

# Copy tiktoken files from builder
COPY --from=builder /tiktoken_cache /root/.cache/huggingface/hub

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "chunk_caption_index_env", "/bin/bash", "-c"]

# Activate conda environment by default
ENV PATH /opt/conda/envs/chunk_caption_index_env/bin:$PATH