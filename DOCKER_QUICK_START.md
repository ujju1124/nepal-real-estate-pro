# 🐳 Docker Quick Start Guide
## Nepal Land & House Price Prediction System

Get your app running in Docker containers in **10 minutes**!

---

## 🎯 What You'll Get

✅ **Containerized app** - Runs anywhere Docker runs  
✅ **No dependency issues** - Everything packaged together  
✅ **Easy deployment** - One command to run  
✅ **Professional setup** - Industry-standard containerization  
✅ **Portfolio boost** - Docker skills for resume  

---

## ⚡ Super Quick Start (3 Commands)

```bash
# 1. Build the Docker image
docker build -t nepal-realestate:latest .

# 2. Run the container
docker run -p 8501:8501 --env-file .env nepal-realestate:latest

# 3. Open your browser
# http://localhost:8501
```

**Done!** Your app is now running in Docker! 🎉

---

## 🛠️ Automated Setup (Recommended)

### For Windows:
```bash
# Build image
docker-scripts\build.bat

# Run container
docker-scripts\run.bat
```

### For Mac/Linux:
```bash
# Make scripts executable
chmod +x docker-scripts/*.sh

# Build image
bash docker-scripts/build.sh

# Run container
bash docker-scripts/run.sh
```

---

## 📋 Step-by-Step Guide

### Step 1: Install Docker

**Windows/Mac:**
- Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Install and start Docker Desktop
- Verify: `docker --version`

**Linux (Ubuntu):**
```bash
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER  # Add yourself to docker group
# Log out and back in
```

### Step 2: Prepare Your Project

**Check required files exist:**
```bash
# These files should exist:
ls -la app_final.py          # ✅ Main app
ls -la requirements.txt      # ✅ Dependencies
ls -la Dockerfile           # ✅ Docker config (created for you)
ls -la *.pkl                # ✅ Model files
```

**Create .env file (for API tokens):**
```bash
# Copy example
cp .env.example .env

# Edit .env and add your tokens:
GITHUB_TOKEN=ghp_your_token_here
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

### Step 3: Build Docker Image

**Option A: Use automated script**
```bash
# Windows
docker-scripts\build.bat

# Mac/Linux
bash docker-scripts/build.sh
```

**Option B: Manual build**
```bash
# Regular build
docker build -t nepal-realestate:latest .

# Optimized build (smaller size)
docker build -f Dockerfile.optimized -t nepal-realestate:optimized .
```

**Build time:** 5-10 minutes (first time)

### Step 4: Run Container

**Option A: Use automated script**
```bash
# Windows
docker-scripts\run.bat

# Mac/Linux
bash docker-scripts/run.sh
```

**Option B: Manual run**
```bash
# With environment file
docker run -p 8501:8501 --env-file .env nepal-realestate:latest

# With individual environment variables
docker run -p 8501:8501 \
  -e GITHUB_TOKEN="your_token" \
  -e HUGGINGFACEHUB_API_TOKEN="your_token" \
  nepal-realestate:latest

# Run in background (detached)
docker run -d -p 8501:8501 \
  --name nepal-realestate-app \
  --env-file .env \
  --restart unless-stopped \
  nepal-realestate:latest
```

### Step 5: Access Your App

**Open browser:** http://localhost:8501

**Test all features:**
- ✅ Analytics section loads
- ✅ Prediction works
- ✅ Recommendations display
- ✅ Chatbot responds (if tokens provided)

---

## 🐙 Docker Compose (Advanced)

**For multi-service setup:**

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

**Services included:**
- Main app (nepal-realestate)
- Optional: Redis (caching)
- Optional: PostgreSQL (database)
- Optional: Nginx (reverse proxy)

---

## 📊 Docker Commands Cheat Sheet

### Building
```bash
# Build image
docker build -t nepal-realestate:latest .

# Build without cache
docker build --no-cache -t nepal-realestate:latest .

# Build optimized version
docker build -f Dockerfile.optimized -t nepal-realestate:optimized .
```

### Running
```bash
# Run interactively (see logs)
docker run -p 8501:8501 nepal-realestate:latest

# Run in background
docker run -d -p 8501:8501 --name myapp nepal-realestate:latest

# Run with environment variables
docker run -p 8501:8501 --env-file .env nepal-realestate:latest
```

### Managing
```bash
# List running containers
docker ps

# List all containers
docker ps -a

# Stop container
docker stop nepal-realestate-app

# Start container
docker start nepal-realestate-app

# Restart container
docker restart nepal-realestate-app

# Remove container
docker rm nepal-realestate-app
```

### Debugging
```bash
# View logs
docker logs nepal-realestate-app

# Follow logs (live)
docker logs -f nepal-realestate-app

# Access container shell
docker exec -it nepal-realestate-app /bin/bash

# Inspect container
docker inspect nepal-realestate-app
```

### Cleanup
```bash
# Remove unused images
docker image prune

# Remove all unused data
docker system prune

# Remove everything (careful!)
docker system prune -a
```

---

## 🔧 Troubleshooting

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
# Use different port
docker run -p 8502:8501 nepal-realestate:latest

# Or find what's using port 8501
# Windows:
netstat -ano | findstr :8501
# Mac/Linux:
lsof -i :8501
```

### Issue 3: "Image build fails"

**Solution:**
```bash
# Check Docker has enough space
docker system df

# Clear cache and rebuild
docker system prune -a
docker build --no-cache -t nepal-realestate:latest .
```

### Issue 4: "Container exits immediately"

**Solution:**
```bash
# Check logs for errors
docker logs nepal-realestate-app

# Run interactively to debug
docker run -it nepal-realestate:latest /bin/bash
```

### Issue 5: "App loads but features don't work"

**Solution:**
```bash
# Check if environment variables are set
docker exec nepal-realestate-app env | grep TOKEN

# Verify model files are in container
docker exec nepal-realestate-app ls -la *.pkl
```

---

## 🚀 Deployment Options

### 1. Your Own Server (VPS)

```bash
# SSH into server
ssh user@your-server-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Clone your repo
git clone https://github.com/YOUR_USERNAME/Nepal-Land-Price-Prediction.git
cd Nepal-Land-Price-Prediction

# Build and run
docker build -t nepal-realestate:latest .
docker run -d -p 8501:8501 \
  --name nepal-realestate-app \
  --restart unless-stopped \
  --env-file .env \
  nepal-realestate:latest

# Access: http://your-server-ip:8501
```

### 2. Cloud Platforms

**Railway:**
- Push to GitHub
- Connect Railway to repo
- Railway auto-detects Dockerfile
- Deploys automatically

**Render:**
- Create `render.yaml` (see Docker guide)
- Push to GitHub
- Connect Render to repo

**Fly.io:**
```bash
fly launch  # Auto-detects Dockerfile
fly deploy
```

---

## 📈 Advantages Summary

### Why Docker for Your Project?

✅ **Consistency:** Works same everywhere  
✅ **Portability:** Deploy to any platform  
✅ **Isolation:** No conflicts with other apps  
✅ **Professionalism:** Industry-standard practice  
✅ **Easy sharing:** Others can run with one command  
✅ **Version control:** Docker images are versioned  
✅ **Scalability:** Easy to run multiple instances  

### Resume/Portfolio Benefits:

- "Containerized ML application using Docker"
- "Implemented CI/CD with Docker"
- "Published Docker image to Docker Hub"
- "Deployed containerized app to cloud platforms"

---

## 📝 For Your Project Report

Add this section:

```markdown
### Containerization

The application is containerized using Docker, enabling consistent 
deployment across different environments.

**Docker Configuration:**
- Base Image: Python 3.11-slim
- Container Size: ~1.2GB (optimized)
- Multi-stage build for size optimization
- Health checks for reliability
- Non-root user for security

**Benefits:**
1. **Portability:** Runs identically on any Docker-enabled platform
2. **Consistency:** Eliminates environment-specific issues
3. **Isolation:** No conflicts with host system
4. **Scalability:** Easy horizontal scaling
5. **Deployment:** Single command deployment

**Usage:**
```bash
docker build -t nepal-realestate .
docker run -p 8501:8501 nepal-realestate
```

This containerization approach demonstrates modern DevOps practices 
and ensures reliable deployment across development, testing, and 
production environments.
```

---

## 🎯 Next Steps

1. **Build your image:** `docker build -t nepal-realestate .`
2. **Test locally:** `docker run -p 8501:8501 nepal-realestate`
3. **Deploy to cloud:** Choose platform and deploy
4. **Add to resume:** "Containerized ML app with Docker"
5. **Share with others:** They can run with one command!

---

## 📞 Need Help?

**Common Commands:**
```bash
# Quick start
docker build -t nepal-realestate . && docker run -p 8501:8501 nepal-realestate

# With environment
docker run -p 8501:8501 --env-file .env nepal-realestate

# Background with restart
docker run -d -p 8501:8501 --name myapp --restart unless-stopped nepal-realestate
```

**Resources:**
- [Docker Documentation](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

**Congratulations!** You now have a professionally containerized ML application! 🐳🎉

**Your app is now:**
- ✅ Containerized with Docker
- ✅ Deployable anywhere
- ✅ Professional and scalable
- ✅ Ready for your portfolio

**Access your app:** http://localhost:8501