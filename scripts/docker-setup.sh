#!/bin/bash

# Godot Docs RAG - Docker Setup Script
set -e

echo "🐳 Godot Docs RAG - Docker Setup"
echo "=================================="

# Color codes for output
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

# Check if Docker and Docker Compose are installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi

    print_success "Docker and Docker Compose are installed"
}

# Setup environment file
setup_env() {
    if [ ! -f ".env" ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env
        print_success ".env file created"
        
        echo ""
        print_warning "⚠️  IMPORTANT: Configure your LLM provider in .env file"
        echo ""
        echo "Choose your LLM provider:"
        echo "1. OpenAI API (requires API key, faster, costs money)"
        echo "2. Local Ollama (free, slower setup, runs locally)"
        echo ""
        
        read -p "Enter your choice (1 for OpenAI, 2 for Ollama): " choice
        
        case $choice in
            1)
                print_status "Configuring for OpenAI..."
                sed -i 's/LLM_PROVIDER=ollama/LLM_PROVIDER=openai/' .env
                echo ""
                print_warning "Please add your OpenAI API key to .env file:"
                print_warning "OPENAI_API_KEY=your_actual_api_key_here"
                echo ""
                read -p "Press Enter after you've added your API key..."
                ;;
            2)
                print_status "Configuring for Ollama (local LLM)..."
                sed -i 's/LLM_PROVIDER=ollama/LLM_PROVIDER=ollama/' .env
                print_success "Ollama configuration set"
                ;;
            *)
                print_warning "Invalid choice, using default Ollama configuration"
                ;;
        esac
    else
        print_success ".env file already exists"
    fi
}

# Create necessary directories
setup_directories() {
    print_status "Creating data directories..."
    mkdir -p data/raw data/parsed data/chunks
    print_success "Directories created"
}

# Build and start services
start_services() {
    print_status "Building and starting Docker services..."
    
    # Build the main application
    docker-compose build
    
    # Start services
    docker-compose up -d ollama
    
    print_status "Waiting for Ollama to be ready..."
    sleep 10
    
    # Check if we're using Ollama and pull models
    if grep -q "LLM_PROVIDER=ollama" .env; then
        print_status "Pulling Ollama models (this may take a few minutes)..."
        docker-compose up ollama-init
        print_success "Ollama models ready"
    fi
    
    print_success "Services started successfully"
}

# Run the main pipeline
run_pipeline() {
    print_status "Running Godot Docs RAG pipeline..."
    docker-compose run --rm godot-docs-rag
    
    if [ $? -eq 0 ]; then
        print_success "Pipeline completed successfully!"
        echo ""
        print_success "📁 Generated Q&A pairs are in: ./data/chunks/"
        print_success "🔍 You can now build your RAG system with this data"
    else
        print_error "Pipeline failed. Check the logs above for details."
        exit 1
    fi
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    docker-compose down
    print_success "Cleanup complete"
}

# Show usage
show_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup     - Setup environment and start services"
    echo "  run       - Run the RAG pipeline"
    echo "  clean     - Stop services and cleanup"
    echo "  logs      - Show service logs"
    echo "  shell     - Open shell in the main container"
    echo ""
    echo "If no command is provided, full setup and run will be executed."
}

# Main execution
main() {
    case "${1:-}" in
        "setup")
            check_docker
            setup_env
            setup_directories
            start_services
            ;;
        "run")
            run_pipeline
            ;;
        "clean")
            cleanup
            ;;
        "logs")
            docker-compose logs -f
            ;;
        "shell")
            docker-compose run --rm godot-docs-rag bash
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        "")
            # Full setup and run
            check_docker
            setup_env
            setup_directories
            start_services
            run_pipeline
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Trap cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"
