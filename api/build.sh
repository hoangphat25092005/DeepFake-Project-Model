#!/bin/bash

# D³ API Docker Build Script
# This script builds the Docker image for the D³ Deepfake Detection API

set -e  # Exit on error

echo "=========================================="
echo "D³ API - Docker Build Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${YELLOW}[INFO]${NC} Script directory: $SCRIPT_DIR"
echo -e "${YELLOW}[INFO]${NC} Project root: $PROJECT_ROOT"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed (V1 or V2)
DOCKER_COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
    echo -e "${GREEN}[OK]${NC} Docker Compose V1 found"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
    echo -e "${GREEN}[OK]${NC} Docker Compose V2 found"
else
    echo -e "${RED}[ERROR]${NC} Docker Compose is not installed."
    echo -e "${YELLOW}[INFO]${NC} Installing Docker Compose V2..."
    echo ""
    echo "Run these commands to install Docker Compose:"
    echo ""
    echo "  # For Ubuntu/Debian:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y docker-compose-plugin"
    echo ""
    echo "  # Or using pip:"
    echo "  pip install docker-compose"
    echo ""
    echo "  # Or download standalone:"
    echo "  sudo curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)\" -o /usr/local/bin/docker-compose"
    echo "  sudo chmod +x /usr/local/bin/docker-compose"
    echo ""
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}[WARNING]${NC} NVIDIA Docker runtime not detected or not working."
    echo -e "${YELLOW}[WARNING]${NC} GPU support may not be available."
    echo ""
fi

# Check if .env file exists
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo -e "${RED}[ERROR]${NC} .env file not found at $SCRIPT_DIR/.env"
    echo -e "${YELLOW}[INFO]${NC} Please create a .env file with required configuration."
    exit 1
fi

# Check if model checkpoint exists
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/finetune_wildrf/model_epoch_best.pth"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${YELLOW}[WARNING]${NC} Model checkpoint not found at:"
    echo -e "${YELLOW}[WARNING]${NC} $CHECKPOINT_PATH"
    echo -e "${YELLOW}[WARNING]${NC} Make sure the model checkpoint is available before running the container."
    echo ""
fi

# Ask for build options
echo -e "${YELLOW}[QUESTION]${NC} Build options:"
echo "  1. Build with cache (faster, recommended for updates)"
echo "  2. Build without cache (clean build, slower)"
read -p "Choose option (1 or 2, default=1): " BUILD_OPTION
BUILD_OPTION=${BUILD_OPTION:-1}

# Build command
cd "$SCRIPT_DIR"

echo ""
echo -e "${GREEN}[BUILD]${NC} Building Docker image..."
echo ""

if [ "$BUILD_OPTION" = "2" ]; then
    echo -e "${YELLOW}[INFO]${NC} Building without cache..."
    $DOCKER_COMPOSE_CMD build --no-cache
else
    echo -e "${YELLOW}[INFO]${NC} Building with cache..."
    $DOCKER_COMPOSE_CMD build
fi

# Check build status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[SUCCESS]${NC} Docker image built successfully!"
    echo ""
    echo "=========================================="
    echo "Next Steps:"
    echo "=========================================="
    echo ""
    echo "1. Start the services:"
    echo "   cd $SCRIPT_DIR"
    echo "   $DOCKER_COMPOSE_CMD up -d"
    echo ""
    echo "2. View logs:"
    echo "   $DOCKER_COMPOSE_CMD logs -f d3_api"
    echo ""
    echo "3. Check status:"
    echo "   $DOCKER_COMPOSE_CMD ps"
    echo ""
    echo "2. View logs:"
    echo "   docker-compose logs -f d3_api"
    echo ""
    echo "3. Check status:"
    echo "   docker-compose ps"
    echo ""
    echo "4. Test the API:"
    echo "   curl http://localhost:5001/health"
    echo ""
    echo "5. Access Swagger UI:"
    echo "   http://localhost:5001/apidocs"
    echo ""
    echo "Or simply run: ./run.sh"
    echo ""
else
    echo ""
    echo -e "${RED}[ERROR]${NC} Docker build failed!"
    echo -e "${YELLOW}[INFO]${NC} Check the error messages above for details."
    exit 1
fi
