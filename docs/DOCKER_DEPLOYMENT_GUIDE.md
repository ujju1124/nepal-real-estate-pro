# Docker Deployment Guide
## Nepal Land & House Price Prediction System

Complete guide to containerizing and deploying your app with Docker.

---

## 🐳 What is Docker?

**Docker** packages your app + all dependencies into a **container** - a lightweight, portable unit that runs the same everywhere.

### Think of it like this:
- **Without Docker:** "It works on my machine!" 🤷
- **With Docker:** "It works everywhere!" ✅

---

## 🎯 Advantages of Using Docker

### 1. **Consistency** ⭐⭐⭐⭐⭐
- Runs identically on your laptop, server, cloud
- No more "works on my machine" problems
- Same Python version, same packages, same everything

### 2. **Portability** ⭐⭐⭐⭐⭐
- Deploy anywhere: AWS, Azure, Google Cloud, your own server
- Easy to move between platforms
- Share with others (they just run `docker run`)

### 3. **Isolation** ⭐⭐⭐⭐
- App runs in its own environment
- Doesn't conflict with other apps
- Clean, reproducible setup

### 4. **Easy Deployment** ⭐⭐⭐⭐
- One command to build: `docker build`
- One command to run: `docker run`
- No manual setup of Python, packages, etc.

### 5. **Version Control** ⭐⭐⭐⭐
- Docker images are versioned
- Easy rollback if something breaks
- Track changes over time

### 6. **Scalability** ⭐⭐⭐
- Easy to run multiple instances
- Load balancing becomes simple
- Kubernetes-ready (for advanced users)

### 7. **Professional Portfolio** ⭐⭐⭐⭐⭐
- Shows you know modern DevOps practices
- Impressive for job interviews
- Industry-standard tool

---

## 📊 Docker vs Traditional Deployment

| Aspect | Traditional | Docker |
|--------|-------------|--------|
| **Setup** | Install Python, packages manually | One `docker build` command |
| **Consistency** | "Works on my machine" | Works everywhere identically |
| **Portability** | Hard to move platforms | Easy to deploy anywhere |
| **Isolation** | Conflicts with other apps | Completely isolated |
| **Deployment** | Complex, error-prone | Simple, reliable |
| **Rollback** | Manual, risky | Easy version control |
| **Learning Curve** | Medium | Medium (but worth it!) |

---

## 🚀 Quick Start: Dockerize Your App

### Prerequisites

**Install Docker:**
- **Windows:** [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- **Mac:** [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- **Linux:** `sudo apt-get install docker.io` (Ubuntu)

**Verify installation:**
```bash
docker --version
# Should show: Docker version 24.x.x
```

---

## 📝 Step 1: Create Dockerfile

Create a file named `Dockerfile` (no extension) in your project root:

```dockerfile
# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies (for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check (optional but recommended)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
ENTRYPOINT ["streamlit", "run", "app_final.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Save this as:** `Dockerfile` (in project root)

---

## 📝 Step 2: Create .dockerignore

Create `.dockerignore` to exclude unnecessary files:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Git
.git/
.gitignore

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Environment
.env

# Documentation (optional - include if you want docs in container)
docs/
*.md

# Data (if large - you might want to download separately)
# *.csv

# Logs
*.log
```

**Save this as:** `.dockerignore`

---

## 🏗️ Step 3: Build Docker Image

```bash
# Build the image
docker build -t nepal-realestate:latest .

# Explanation:
# -t nepal-realestate:latest = Tag the image with name and version
# . = Build context (current directory)

# This will take 5-10 minutes first time
```

**What happens during build:**
1. Downloads Python 3.11 base image
2. Installs system dependencies
3. Installs Python packages from requirements.txt
4. Copies your app files
5. Sets up Streamlit to run

**Check your image:**
```bash
docker images

# You should see:
# REPOSITORY          TAG       SIZE
# nepal-realestate    latest    ~1.5GB
```

---

## 🚀 Step 4: Run Docker Container

### Basic Run (No Secrets):

```bash
docker run -p 8501:8501 nepal-realestate:latest

# Explanation:
# -p 8501:8501 = Map port 8501 (container) to 8501 (host)
# nepal-realestate:latest = Image to run
```

**Access app:** http://localhost:8501

### Run with Environment Variables (Secrets):

```bash
docker run -p 8501:8501 \
  -e GITHUB_TOKEN="ghp_your_token_here" \
  -e HUGGINGFACEHUB_API_TOKEN="hf_your_token_here" \
  nepal-realestate:latest
```

### Run with .env File:

```bash
docker run -p 8501:8501 \
  --env-file .env \
  nepal-realestate:latest
```

### Run in Background (Detached):

```bash
docker run -d -p 8501:8501 \
  --name nepal-realestate-app \
  --env-file .env \
  nepal-realestate:latest

# -d = Detached mode (runs in background)
# --name = Give container a name
```

**Check running containers:**
```bash
docker ps

# Stop container:
docker stop nepal-realestate-app

# Start again:
docker start nepal-realestate-app

# View logs:
docker logs nepal-realestate-app

# Follow logs (live):
docker logs -f nepal-realestate-app
```

---

## 🎨 Step 5: Optimize Docker Image (Advanced)

### Multi-Stage Build (Reduce Size)

Create `Dockerfile.optimized`:

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app_final.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build optimized:**
```bash
docker build -f Dockerfile.optimized -t nepal-realestate:optimized .
```

**Size comparison:**
- Regular: ~1.5GB
- Optimized: ~1.2GB (20% smaller)

---

## 🌐 Step 6: Deploy Docker Container

### Option 1: Deploy to Your Own Server (VPS)

**Get a free VPS:**
- Oracle Cloud: Free tier (1-2 VMs forever)
- Google Cloud: $300 credit (12 months)
- AWS: Free tier (12 months)

**Deploy steps:**

```bash
# 1. SSH into your server
ssh user@your-server-ip

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 3. Clone your repo
git clone https://github.com/YOUR_USERNAME/Nepal-Land-Price-Prediction.git
cd Nepal-Land-Price-Prediction

# 4. Build image
docker build -t nepal-realestate:latest .

# 5. Run container
docker run -d -p 8501:8501 \
  --name nepal-realestate-app \
  --restart unless-stopped \
  -e GITHUB_TOKEN="your_token" \
  -e HUGGINGFACEHUB_API_TOKEN="your_token" \
  nepal-realestate:latest

# 6. Access app
# http://your-server-ip:8501
```

**Auto-restart on reboot:**
```bash
docker update --restart unless-stopped nepal-realestate-app
```

---

### Option 2: Deploy to Render (with Docker)

**Render supports Docker deployments:**

1. Create `render.yaml`:

```yaml
services:
  - type: web
    name: nepal-realestate
    env: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: GITHUB_TOKEN
        sync: false
      - key: HUGGINGFACEHUB_API_TOKEN
        sync: false
```

2. Push to GitHub
3. Connect Render to your repo
4. Render auto-deploys from Dockerfile

**Advantages:**
- Free tier available
- Auto-deploys on git push
- HTTPS included
- Custom domain support

---

### Option 3: Deploy to Railway (with Docker)

**Railway auto-detects Dockerfile:**

1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub"
3. Select your repo
4. Railway detects Dockerfile and builds automatically
5. Add environment variables in dashboard
6. Get public URL

**Cost:** $5 credit/month (free)

---

### Option 4: Deploy to Fly.io (Free Tier)

**Fly.io offers free Docker hosting:**

```bash
# 1. Install flyctl
curl -L https://fly.io/install.sh | sh

# 2. Sign up
fly auth signup

# 3. Launch app
fly launch

# Follow prompts:
# - App name: nepal-realestate
# - Region: Choose closest to you
# - Database: No

# 4. Set secrets
fly secrets set GITHUB_TOKEN="your_token"
fly secrets set HUGGINGFACEHUB_API_TOKEN="your_token"

# 5. Deploy
fly deploy

# 6. Open app
fly open
```

**Free tier:**
- 3 shared-cpu VMs
- 256MB RAM each
- 3GB storage

---

## 🐙 Step 7: Docker Compose (Multiple Services)

If you want to add a database or other services:

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
    volumes:
      - ./data:/app/data  # Mount data directory
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Add Redis for caching
  # redis:
  #   image: redis:alpine
  #   ports:
  #     - "6379:6379"
  #   restart: unless-stopped

  # Optional: Add PostgreSQL for data storage
  # postgres:
  #   image: postgres:15-alpine
  #   environment:
  #     POSTGRES_PASSWORD: your_password
  #   volumes:
  #     - postgres_data:/var/lib/postgresql/data
  #   restart: unless-stopped

# volumes:
#   postgres_data:
```

**Run with Docker Compose:**
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild and restart
docker-compose up -d --build
```

---

## 📦 Step 8: Push to Docker Hub (Share Your Image)

**Docker Hub = GitHub for Docker images**

```bash
# 1. Create account at hub.docker.com

# 2. Login
docker login

# 3. Tag your image
docker tag nepal-realestate:latest YOUR_USERNAME/nepal-realestate:latest

# 4. Push to Docker Hub
docker push YOUR_USERNAME/nepal-realestate:latest

# Now anyone can run your app:
docker run -p 8501:8501 YOUR_USERNAME/nepal-realestate:latest
```

**Add to your resume:**
- "Containerized ML application using Docker"
- "Published Docker image to Docker Hub"
- "Implemented CI/CD with Docker"

---

## 🎓 For Your Project Report

Add this section:

```markdown
### Containerization with Docker

The application is containerized using Docker, enabling consistent 
deployment across different environments and platforms.

**Docker Configuration:**
- Base Image: Python 3.11-slim
- Container Size: ~1.2GB (optimized)
- Exposed Port: 8501
- Health Checks: Enabled
- Auto-restart: Configured

**Dockerfile Structure:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app_final.py"]
```

**Benefits:**
1. **Portability:** Runs identically on any platform
2. **Consistency:** Eliminates "works on my machine" issues
3. **Isolation:** No conflicts with other applications
4. **Scalability:** Easy to deploy multiple instances
5. **Version Control:** Docker images are versioned and reproducible

**Deployment Options:**
- Local development: `docker run -p 8501:8501 nepal-realestate`
- Cloud platforms: AWS ECS, Google Cloud Run, Azure Container Instances
- Kubernetes: For production-scale deployments

This containerization approach demonstrates modern DevOps practices 
and ensures the application can be deployed reliably in any environment.
```

---

## 🔧 Troubleshooting Docker

### Issue 1: "Cannot connect to Docker daemon"

**Solution:**
```bash
# Windows/Mac: Start Docker Desktop
# Linux: Start Docker service
sudo systemctl start docker
```

### Issue 2: "Port 8501 already in use"

**Solution:**
```bash
# Find what's using port 8501
# Windows:
netstat -ano | findstr :8501

# Mac/Linux:
lsof -i :8501

# Kill the process or use different port:
docker run -p 8502:8501 nepal-realestate:latest
```

### Issue 3: "Image build fails"

**Solution:**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t nepal-realestate:latest .
```

### Issue 4: "Container exits immediately"

**Solution:**
```bash
# Check logs
docker logs nepal-realestate-app

# Run interactively to see errors
docker run -it nepal-realestate:latest /bin/bash
```

---

## 📊 Docker Commands Cheat Sheet

```bash
# BUILD
docker build -t nepal-realestate:latest .
docker build --no-cache -t nepal-realestate:latest .

# RUN
docker run -p 8501:8501 nepal-realestate:latest
docker run -d -p 8501:8501 --name myapp nepal-realestate:latest
docker run -it nepal-realestate:latest /bin/bash  # Interactive

# MANAGE
docker ps                    # List running containers
docker ps -a                 # List all containers
docker stop myapp            # Stop container
docker start myapp           # Start container
docker restart myapp         # Restart container
docker rm myapp              # Remove container

# IMAGES
docker images                # List images
docker rmi nepal-realestate  # Remove image
docker pull python:3.11      # Download image

# LOGS
docker logs myapp            # View logs
docker logs -f myapp         # Follow logs (live)
docker logs --tail 100 myapp # Last 100 lines

# CLEANUP
docker system prune          # Remove unused data
docker system prune -a       # Remove all unused images
docker volume prune          # Remove unused volumes

# INSPECT
docker inspect myapp         # Detailed container info
docker stats                 # Resource usage
docker exec -it myapp /bin/bash  # Access running container
```

---

## 🎯 Summary: Why Use Docker?

### For Your Project:

✅ **Professional:** Shows modern DevOps skills  
✅ **Portable:** Deploy anywhere (AWS, Azure, your laptop)  
✅ **Consistent:** No "works on my machine" issues  
✅ **Resume Boost:** Docker is industry-standard  
✅ **Easy Sharing:** Others can run with one command  
✅ **Scalable:** Easy to run multiple instances  

### When to Use Docker:

- ✅ Deploying to your own server
- ✅ Need consistent environment
- ✅ Want to learn DevOps
- ✅ Planning to scale later
- ✅ Want professional portfolio piece

### When NOT to Use Docker:

- ❌ Just need quick demo (use Streamlit Cloud)
- ❌ Don't have time to learn
- ❌ Only deploying to Streamlit Cloud (not needed)

---

## 🚀 Next Steps

1. **Create Dockerfile** (copy from above)
2. **Build image:** `docker build -t nepal-realestate .`
3. **Test locally:** `docker run -p 8501:8501 nepal-realestate`
4. **Deploy** to your chosen platform
5. **Add to resume:** "Containerized ML app with Docker"

---

**Congratulations!** You now know how to containerize and deploy your ML app with Docker! 🐳
