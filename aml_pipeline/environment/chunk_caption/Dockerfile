# Use the Azure ML base image
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# Copy conda environment file
COPY conda.yaml /tmp/conda.yaml

# Create conda environment
RUN conda env create -f /tmp/conda.yaml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "chunk_caption_index_env", "/bin/bash", "-c"]

# Download tiktoken files during build
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Activate conda environment by default
ENV PATH /opt/conda/envs/chunk_caption_index_env/bin:$PATH