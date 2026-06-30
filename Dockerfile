# ═══════════════════════════════════════════════════════════════
# Nepal Real Estate Pro — Hugging Face Spaces Dockerfile
# ═══════════════════════════════════════════════════════════════
# Optimized for HF Spaces free CPU tier (limited RAM)
# Port: 7860, listens on 0.0.0.0

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by scientific packages
# - build-essential: for compiling some Python packages
# - libgomp1: required by LightGBM and some ML libraries
# - git: for some Python packages that install from git
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app_final.py .
COPY .env.example .env
COPY data/ ./data/
COPY models/ ./models/

# Create .streamlit directory for config
RUN mkdir -p /root/.streamlit

# Create Streamlit config for HF Spaces
RUN echo '\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
port = 7860\n\
address = "0.0.0.0"\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /root/.streamlit/config.toml

# Expose port 7860 for HF Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Run Streamlit app with HF Spaces configuration
CMD ["streamlit", "run", "app_final.py", \
     "--server.address=0.0.0.0", \
     "--server.port=7860", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--server.headless=true"]
