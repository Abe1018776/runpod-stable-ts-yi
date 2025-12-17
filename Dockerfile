# Base image
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Define your working directory
WORKDIR /app

# Configure LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"

# Install relevant packages 
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir \
    ivrit[all]==0.1.8 \
    torch==2.4.1 \
    huggingface-hub==0.36.0 \
    runpod \
    scipy \
    yt-dlp \
    faster-whisper \
    pyannote.audio \
    speechbrain \
    git+https://github.com/ivrit-ai/stable-ts.git

# Pass HF_TOKEN build argument to environment
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Copy builder script and bake the model into the image
# This ensures faster cold starts by pre-downloading the large model
COPY builder.py .
RUN python builder.py

# Copy application source code
COPY src/ .

# Command to start the RunPod handler
CMD [ "python", "-u", "handler.py" ]
