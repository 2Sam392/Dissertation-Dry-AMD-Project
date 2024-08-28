# Stage 1: Download model weights
FROM python:3.8-slim as downloader

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install wget
RUN apt-get update && apt-get install -y wget && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a directory to store the model weights
RUN mkdir -p /weights

# Download model weights
RUN wget -O /weights/vgg16-397923af.pth https://download.pytorch.org/models/vgg16-397923af.pth
RUN wget -O /weights/resnet50-0676ba61.pth https://download.pytorch.org/models/resnet50-0676ba61.pth

# Stage 2: Build final image
FROM python:3.8-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  python3-setuptools \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install torch torchvision Pillow streamlit

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Copy the downloaded model weights from the downloader stage
COPY --from=downloader /weights /root/.cache/torch/hub/checkpoints

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit application
CMD ["streamlit", "run", "app.py"]
