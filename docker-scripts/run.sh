#!/bin/bash
# Run script for Nepal Real Estate Docker container

set -e  # Exit on any error

# Configuration
IMAGE_NAME="nepal-realestate"
TAG="latest"
CONTAINER_NAME="nepal-realestate-app"
PORT="8501"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🚀 Running Nepal Real Estate Docker Container..."
echo "================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if image exists
if ! docker images "${IMAGE_NAME}:${TAG}" --format "{{.Repository}}" | grep -q "${IMAGE_NAME}"; then
    print_error "Docker image '${IMAGE_NAME}:${TAG}' not found!"
    print_status "Please build the image first:"
    echo "  bash docker-scripts/build.sh"
    echo "  OR"
    echo "  docker build -t ${IMAGE_NAME}:${TAG} ."
    exit 1
fi

print_success "Docker image found ✓"

# Stop existing container if running
if docker ps -q -f name="${CONTAINER_NAME}" | grep -q .; then
    print_warning "Stopping existing container..."
    docker stop "${CONTAINER_NAME}" > /dev/null
fi

# Remove existing container if exists
if docker ps -aq -f name="${CONTAINER_NAME}" | grep -q .; then
    print_warning "Removing existing container..."
    docker rm "${CONTAINER_NAME}" > /dev/null
fi

# Check if .env file exists
if [[ -f ".env" ]]; then
    print_success ".env file found - will use environment variables ✓"
    ENV_FILE_OPTION="--env-file .env"
else
    print_warning ".env file not found"
    print_status "Container will run without API tokens (RAG chatbot won't work)"
    ENV_FILE_OPTION=""
fi

# Check if port is available
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_error "Port ${PORT} is already in use!"
    print_status "Please stop the service using port ${PORT} or use a different port:"
    echo "  docker run -p 8502:8501 ${ENV_FILE_OPTION} ${IMAGE_NAME}:${TAG}"
    exit 1
fi

print_success "Port ${PORT} is available ✓"

# Run container
print_status "Starting container..."
echo "Command: docker run -d -p ${PORT}:8501 --name ${CONTAINER_NAME} ${ENV_FILE_OPTION} --restart unless-stopped ${IMAGE_NAME}:${TAG}"
echo ""

if docker run -d \
    -p "${PORT}:8501" \
    --name "${CONTAINER_NAME}" \
    ${ENV_FILE_OPTION} \
    --restart unless-stopped \
    "${IMAGE_NAME}:${TAG}"; then
    
    print_success "Container started successfully!"
    
    # Wait a moment for container to start
    sleep 3
    
    # Check if container is running
    if docker ps -q -f name="${CONTAINER_NAME}" | grep -q .; then
        print_success "Container is running ✓"
        
        # Show container info
        print_status "Container details:"
        docker ps -f name="${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        
        echo ""
        print_status "🎉 Your app is now running!"
        echo ""
        echo "📱 Access your app:"
        echo "   Local:    http://localhost:${PORT}"
        echo "   Network:  http://$(hostname -I | awk '{print $1}'):${PORT}"
        echo ""
        print_status "📊 Useful commands:"
        echo "   View logs:     docker logs -f ${CONTAINER_NAME}"
        echo "   Stop app:      docker stop ${CONTAINER_NAME}"
        echo "   Start app:     docker start ${CONTAINER_NAME}"
        echo "   Restart app:   docker restart ${CONTAINER_NAME}"
        echo "   Remove app:    docker rm -f ${CONTAINER_NAME}"
        echo ""
        
        # Show logs for a few seconds
        print_status "📋 Recent logs (last 10 lines):"
        docker logs --tail 10 "${CONTAINER_NAME}"
        
        echo ""
        print_status "💡 To follow logs in real-time:"
        echo "   docker logs -f ${CONTAINER_NAME}"
        
    else
        print_error "Container failed to start!"
        print_status "Checking logs..."
        docker logs "${CONTAINER_NAME}"
        exit 1
    fi
    
else
    print_error "Failed to start container!"
    exit 1
fi

echo ""
echo "================================================"
print_success "Container is running successfully! 🎉"