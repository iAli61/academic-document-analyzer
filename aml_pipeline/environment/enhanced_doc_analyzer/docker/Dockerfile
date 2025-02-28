FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

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