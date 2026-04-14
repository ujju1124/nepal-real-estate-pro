# Nepal Land & House Price Prediction System
# Docker configuration for production deployment

# Use official Python runtime as base image
FROM python:3.11-slim

# Set metadata
LABEL maintainer="Ujju"
LABEL description="Nepal Real Estate Price Prediction System"
LABEL version="1.0"

# Set working directory in container
WORKDIR /app

# Install system dependencies
# - build-essential: For compiling Python packages
# - curl: For health checks
# - software-properties-common: For adding repositories
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
# If requirements.txt doesn't change, this layer is cached
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Don't cache pip packages (reduces image size)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for Streamlit config (if not exists)
RUN mkdir -p /root/.streamlit

# Copy Streamlit config
COPY .streamlit/config.toml /root/.streamlit/config.toml

# Expose Streamlit default port
EXPOSE 8501

# Health check - ensures container is healthy
# Checks every 30 seconds if Streamlit is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the application
# Using ENTRYPOINT + CMD allows overriding CMD at runtime
ENTRYPOINT ["streamlit", "run"]
CMD ["app_final.py", "--server.port=8501", "--server.address=0.0.0.0"]
