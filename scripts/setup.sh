#!/bin/bash

# Godot RAG System Setup Script
# Automated installation and configuration

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.8"
OLLAMA_MODELS=("llama3.2" "nomic-embed-text")
QDRANT_VERSION="v1.7.4"

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to compare version numbers
version_compare() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Function to check Python version
check_python() {
    print_status "Checking Python installation..."
    
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed. Please install Python $PYTHON_MIN_VERSION or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_status "Found Python $PYTHON_VERSION"
    
    if ! version_compare "$PYTHON_VERSION" "$PYTHON_MIN_VERSION"; then
        print_error "Python $PYTHON_MIN_VERSION or higher is required. Found $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python version check passed"
}

# Function to create virtual environment
setup_virtual_env() {
    print_status "Setting up Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Do you want to recreate it? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -rf venv
        else
            print_status "Using existing virtual environment"
            return 0
        fi
    fi
    
    $PYTHON_CMD -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    print_success "Virtual environment created and activated"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Make sure we're in the virtual environment
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source venv/bin/activate
    fi
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Install development dependencies if requested
    if [ -f "requirements-dev.txt" ]; then
        print_status "Install development dependencies? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            pip install -r requirements-dev.txt
            print_success "Development dependencies installed"
        fi
    fi
}

# Function to check and install Docker
setup_docker() {
    print_status "Checking Docker installation..."
    
    if command_exists docker && command_exists docker-compose; then
        print_success "Docker and Docker Compose are installed"
        
        # Check if user is in docker group
        if groups $USER | grep -q '\bdocker\b'; then
            print_success "User is in docker group"
        else
            print_warning "User is not in docker group. You may need to run docker commands with sudo."
            print_status "Add user to docker group? (requires logout/login) (y/N)"
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                sudo usermod -aG docker $USER
                print_success "User added to docker group. Please logout and login again."
            fi
        fi
    else
        print_warning "Docker is not installed. Install Docker? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            # Install Docker
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
            
            # Install Docker Compose
            sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            sudo chmod +x /usr/local/bin/docker-compose
            
            print_success "Docker installed. Please logout and login again."
        else
            print_warning "Skipping Docker installation. Some features may not work."
        fi
    fi
}

# Function to setup Ollama
setup_ollama() {
    print_status "Setting up Ollama..."
    
    if command_exists ollama; then
        print_success "Ollama is already installed"
    else
        print_status "Install Ollama? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            curl -fsSL https://ollama.ai/install.sh | sh
            print_success "Ollama installed"
        else
            print_warning "Skipping Ollama installation. LLM features may not work."
            return 0
        fi
    fi
    
    # Start Ollama service
    print_status "Starting Ollama service..."
    if systemctl is-active --quiet ollama; then
        print_success "Ollama service is running"
    else
        if command_exists systemctl; then
            sudo systemctl start ollama
            sudo systemctl enable ollama
            print_success "Ollama service started"
        else
            print_status "Starting Ollama in the background..."
            ollama serve &
            sleep 5
        fi
    fi
    
    # Pull required models
    print_status "Downloading required models..."
    for model in "${OLLAMA_MODELS[@]}"; do
        print_status "Pulling model: $model"
        if ollama pull "$model"; then
            print_success "Model $model downloaded"
        else
            print_error "Failed to download model $model"
        fi
    done
}

# Function to setup Qdrant
setup_qdrant() {
    print_status "Setting up Qdrant vector database..."
    
    if [ -f "docker-compose.yml" ]; then
        print_status "Start Qdrant with Docker Compose? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            docker-compose up -d qdrant
            print_success "Qdrant started via Docker Compose"
        fi
    else
        print_status "Run Qdrant in Docker? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            docker run -d \
                --name qdrant \
                -p 6333:6333 \
                -p 6334:6334 \
                -v $(pwd)/data/qdrant:/qdrant/storage \
                qdrant/qdrant:$QDRANT_VERSION
            print_success "Qdrant started in Docker"
        fi
    fi
}

# Function to create directories
create_directories() {
    print_status "Creating project directories..."
    
    directories=(
        "data/raw"
        "data/processed"
        "data/chunks"
        "data/qdrant"
        "logs"
        "metrics"
        "config"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    print_success "Project directories created"
}

# Function to create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    if [ -f ".env" ]; then
        print_warning ".env file already exists. Overwrite? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    cat > .env << EOF
# Godot RAG System Configuration

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
EMBEDDING_MODEL=nomic-embed-text

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=godot_docs
EMBEDDING_DIMENSION=768

# Data Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
HYBRID_ALPHA=0.7

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Streamlit Configuration
STREAMLIT_PORT=8501

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/godot_rag.log

# Development
DEBUG=false
ENVIRONMENT=development
EOF
    
    print_success "Environment file created (.env)"
}

# Function to run initial data processing
run_initial_processing() {
    print_status "Run initial data processing? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_status "Processing Godot documentation..."
        
        # Make sure we're in the virtual environment
        if [[ "$VIRTUAL_ENV" == "" ]]; then
            source venv/bin/activate
        fi
        
        # Run data processing
        python -c "
import asyncio
from src.data_processor import DocumentProcessor
from pathlib import Path

async def main():
    processor = DocumentProcessor(Path('data'))
    stats = await processor.process_all_sources()
    print(f'Processing completed: {stats.successful_documents}/{stats.total_documents} documents')

if __name__ == '__main__':
    asyncio.run(main())
"
        print_success "Initial data processing completed"
    fi
}

# Function to create startup scripts
create_startup_scripts() {
    print_status "Creating startup scripts..."
    
    # Create start script for FastAPI
    cat > start_api.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
EOF
    chmod +x start_api.sh
    
    # Create start script for Streamlit
    cat > start_ui.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
EOF
    chmod +x start_ui.sh
    
    # Create development script
    cat > start_dev.sh << 'EOF'
#!/bin/bash
echo "Starting Godot RAG System in development mode..."

# Start services
if command -v docker-compose &> /dev/null; then
    echo "Starting Docker services..."
    docker-compose up -d
fi

# Start API in background
echo "Starting FastAPI server..."
./start_api.sh &
API_PID=$!

# Start Streamlit
echo "Starting Streamlit UI..."
./start_ui.sh &
UI_PID=$!

echo "Services started:"
echo "- FastAPI: http://localhost:8000"
echo "- Streamlit UI: http://localhost:8501"
echo "- API Docs: http://localhost:8000/docs"

# Function to cleanup on exit
cleanup() {
    echo "Shutting down services..."
    kill $API_PID $UI_PID 2>/dev/null
    if command -v docker-compose &> /dev/null; then
        docker-compose down
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Press Ctrl+C to stop all services"
wait
EOF
    chmod +x start_dev.sh
    
    print_success "Startup scripts created"
}

# Function to run health checks
run_health_checks() {
    print_status "Running health checks..."
    
    # Check if Ollama is responding
    if curl -s http://localhost:11434/api/tags >/dev/null; then
        print_success "✓ Ollama API is responding"
    else
        print_warning "✗ Ollama API is not responding"
    fi
    
    # Check if Qdrant is responding
    if curl -s http://localhost:6333/health >/dev/null; then
        print_success "✓ Qdrant is responding"
    else
        print_warning "✗ Qdrant is not responding"
    fi
    
    # Check Python environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_success "✓ Virtual environment is active"
    else
        print_warning "✗ Virtual environment is not active"
    fi
    
    # Check if all models are available
    for model in "${OLLAMA_MODELS[@]}"; do
        if ollama list | grep -q "$model"; then
            print_success "✓ Model $model is available"
        else
            print_warning "✗ Model $model is not available"
        fi
    done
}

# Main setup function
main() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Godot RAG System Setup${NC}"
    echo -e "${BLUE}================================${NC}"
    echo
    
    print_status "This script will set up the Godot RAG system with all dependencies."
    print_status "Continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
    
    # Run setup steps
    check_python
    setup_virtual_env
    install_python_deps
    setup_docker
    setup_ollama
    setup_qdrant
    create_directories
    create_env_file
    create_startup_scripts
    run_initial_processing
    run_health_checks
    
    echo
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}  Setup Complete!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo
    print_success "Godot RAG System is ready to use!"
    echo
    echo "Next steps:"
    echo "1. Start the system: ./start_dev.sh"
    echo "2. Open the web interface: http://localhost:8501"
    echo "3. Check the API documentation: http://localhost:8000/docs"
    echo
    echo "For individual services:"
    echo "- API only: ./start_api.sh"
    echo "- UI only: ./start_ui.sh"
    echo
    print_status "Enjoy using the Godot RAG Assistant! 🎮"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
