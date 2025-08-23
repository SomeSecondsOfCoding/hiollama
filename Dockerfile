# Base image with Ollama preinstalled
FROM ghcr.io/ollama/ollama:latest

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy app + assets + deps
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
COPY assets /app/assets
RUN mkdir -p /app/data

# (Optional) Pre-pull models to speed up first run
RUN ollama serve & sleep 5 && \
    ollama pull tinyllama && \
    ollama pull nomic-embed-text && \
    pkill ollama || true

# Expose Streamlit port
EXPOSE 7860

# Run both Ollama and Streamlit
CMD ollama serve & \
    streamlit run app.py --server.port 7860 --server.address 0.0.0.0
