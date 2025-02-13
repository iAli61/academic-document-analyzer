# Use the Azure ML base image
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# Copy conda environment file
COPY conda.yaml /tmp/conda.yaml

# Copy pre-downloaded tiktoken files
COPY tiktoken_files/* /root/.cache/tiktoken/

# Create conda environment
RUN conda env create -f /tmp/conda.yaml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "chunk_caption_index_env", "/bin/bash", "-c"]

# Activate conda environment by default
ENV PATH /opt/conda/envs/chunk_caption_index_env/bin:$PATH