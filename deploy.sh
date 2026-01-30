#!/bin/bash
# Manga Translator - Quick Deploy Script
# Usage: ./deploy.sh [option]
#   Options: install, run, docker, stop

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# ==============================================================================
# Install dependencies
# ==============================================================================
install_deps() {
    echo "=================================="
    echo "Installing Manga Translator"
    echo "=================================="

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3.10+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    print_status "Python version: $PYTHON_VERSION"

    # Create virtual environment if not exists
    if [ ! -d ".venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv .venv
    fi

    # Activate virtual environment
    source .venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install dependencies
    print_status "Installing dependencies..."
    pip install -r requirements.txt

    # Download models
    print_status "Checking models..."
    python download_models.py status

    print_status "Installation complete!"
    echo ""
    echo "To activate the environment:"
    echo "  source .venv/bin/activate"
    echo ""
    echo "To run the server:"
    echo "  python app.py"
}

# ==============================================================================
# Run development server
# ==============================================================================
run_dev() {
    echo "=================================="
    echo "Starting Development Server"
    echo "=================================="

    # Activate virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    # Check if models exist
    if [ ! -f "model/model.pt" ]; then
        print_warning "Model not found. Please ensure model/model.pt exists."
    fi

    # Run Flask app
    print_status "Starting Flask server on http://localhost:5000"
    python app.py
}

# ==============================================================================
# Run production server
# ==============================================================================
run_prod() {
    echo "=================================="
    echo "Starting Production Server"
    echo "=================================="

    # Activate virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    # Check gunicorn
    if ! command -v gunicorn &> /dev/null; then
        print_status "Installing gunicorn..."
        pip install gunicorn
    fi

    # Set environment variables
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4

    # Preload models
    print_status "Preloading models..."
    python -c "from services import Services; Services.preload_all()" || true

    # Run with gunicorn
    print_status "Starting Gunicorn server on http://0.0.0.0:5000"
    gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
}

# ==============================================================================
# Docker build and run
# ==============================================================================
docker_run() {
    echo "=================================="
    echo "Docker Deployment"
    echo "=================================="

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker."
        exit 1
    fi

    # Build image
    print_status "Building Docker image..."
    docker build -t manga-translator .

    # Stop existing container
    docker stop manga-translator 2>/dev/null || true
    docker rm manga-translator 2>/dev/null || true

    # Run container
    print_status "Starting container..."
    docker run -d \
        --name manga-translator \
        -p 5000:5000 \
        -v "$(pwd)/model:/app/model" \
        -v "$(pwd)/fonts:/app/fonts" \
        --restart unless-stopped \
        manga-translator

    print_status "Container started!"
    echo ""
    echo "API available at: http://localhost:5000"
    echo ""
    echo "View logs: docker logs -f manga-translator"
    echo "Stop: docker stop manga-translator"
}

# ==============================================================================
# Docker Compose
# ==============================================================================
docker_compose_up() {
    echo "=================================="
    echo "Docker Compose Deployment"
    echo "=================================="

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose not found."
        exit 1
    fi

    # Use docker compose or docker-compose
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi

    print_status "Starting services..."
    $COMPOSE_CMD up -d --build

    print_status "Services started!"
    echo ""
    echo "API available at: http://localhost:5000"
    echo ""
    echo "View logs: $COMPOSE_CMD logs -f"
    echo "Stop: $COMPOSE_CMD down"
}

# ==============================================================================
# Stop all services
# ==============================================================================
stop_all() {
    echo "=================================="
    echo "Stopping All Services"
    echo "=================================="

    # Stop Docker container
    docker stop manga-translator 2>/dev/null && print_status "Stopped Docker container" || true

    # Stop Docker Compose
    if docker compose version &> /dev/null; then
        docker compose down 2>/dev/null && print_status "Stopped Docker Compose" || true
    elif command -v docker-compose &> /dev/null; then
        docker-compose down 2>/dev/null && print_status "Stopped Docker Compose" || true
    fi

    # Kill gunicorn
    pkill -f "gunicorn.*app:app" 2>/dev/null && print_status "Stopped Gunicorn" || true

    # Kill Flask dev server
    pkill -f "python app.py" 2>/dev/null && print_status "Stopped Flask dev server" || true

    print_status "All services stopped"
}

# ==============================================================================
# Show help
# ==============================================================================
show_help() {
    echo "Manga Translator - Deploy Script"
    echo ""
    echo "Usage: ./deploy.sh [command]"
    echo ""
    echo "Commands:"
    echo "  install     Install dependencies and setup environment"
    echo "  dev         Run development server (Flask)"
    echo "  prod        Run production server (Gunicorn)"
    echo "  docker      Build and run Docker container"
    echo "  compose     Run with Docker Compose"
    echo "  stop        Stop all running services"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh install    # First time setup"
    echo "  ./deploy.sh dev        # Development"
    echo "  ./deploy.sh prod       # Production"
    echo "  ./deploy.sh docker     # Docker deployment"
}

# ==============================================================================
# Main
# ==============================================================================
case "${1:-help}" in
    install)
        install_deps
        ;;
    dev|run)
        run_dev
        ;;
    prod|production)
        run_prod
        ;;
    docker)
        docker_run
        ;;
    compose|docker-compose)
        docker_compose_up
        ;;
    stop)
        stop_all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
