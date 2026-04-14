#!/bin/bash
# Build script for Nepal Real Estate Docker image

set -e  # Exit on any error

echo "🐳 Building Nepal Real Estate Docker Image..."
echo "================================================"

# Configuration
IMAGE_NAME="nepal-realestate"
TAG="latest"
OPTIMIZED_TAG="optimized"

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_success "Docker is running ✓"

# Check if required files exist
required_files=("app_final.py" "requirements.txt" "Dockerfile")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        print_error "Required file '$file' not found!"
        exit 1
    fi
done

print_success "All required files found ✓"

# Check if model files exist
model_files=(
    "xgboost_housing_final.pkl"
    "catboost_land_model_final.pkl"
    "catboost_lalpurja_house_v2_final.pkl"
    "catboost_lalpurja_model_final.pkl"
    "scaler_lalpurja_house_v2.pkl"
)

missing_models=()
for model in "${model_files[@]}"; do
    if [[ ! -f "$model" ]]; then
        missing_models+=("$model")
    fi
done

if [[ ${#missing_models[@]} -gt 0 ]]; then
    print_warning "Missing model files:"
    for model in "${missing_models[@]}"; do
        echo "  - $model"
    done
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Build cancelled."
        exit 1
    fi
else
    print_success "All model files found ✓"
fi

# Build regular image
print_status "Building regular Docker image..."
echo "Command: docker build -t ${IMAGE_NAME}:${TAG} ."
echo ""

if docker build -t "${IMAGE_NAME}:${TAG}" .; then
    print_success "Regular image built successfully!"
    
    # Get image size
    size=$(docker images "${IMAGE_NAME}:${TAG}" --format "table {{.Size}}" | tail -n 1)
    print_status "Image size: $size"
else
    print_error "Failed to build regular image"
    exit 1
fi

echo ""

# Ask if user wants to build optimized version
read -p "Build optimized version? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Building optimized Docker image..."
    echo "Command: docker build -f Dockerfile.optimized -t ${IMAGE_NAME}:${OPTIMIZED_TAG} ."
    echo ""
    
    if docker build -f Dockerfile.optimized -t "${IMAGE_NAME}:${OPTIMIZED_TAG}" .; then
        print_success "Optimized image built successfully!"
        
        # Get optimized image size
        opt_size=$(docker images "${IMAGE_NAME}:${OPTIMIZED_TAG}" --format "table {{.Size}}" | tail -n 1)
        print_status "Optimized image size: $opt_size"
    else
        print_error "Failed to build optimized image"
    fi
fi

echo ""
echo "================================================"
print_success "Build completed!"
echo ""

# Show built images
print_status "Built images:"
docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

echo ""
print_status "Next steps:"
echo "1. Test locally:"
echo "   docker run -p 8501:8501 ${IMAGE_NAME}:${TAG}"
echo ""
echo "2. Run with environment variables:"
echo "   docker run -p 8501:8501 --env-file .env ${IMAGE_NAME}:${TAG}"
echo ""
echo "3. Run in background:"
echo "   docker run -d -p 8501:8501 --name nepal-realestate-app ${IMAGE_NAME}:${TAG}"
echo ""
echo "4. Use Docker Compose:"
echo "   docker-compose up -d"
echo ""
print_status "Access your app at: http://localhost:8501"