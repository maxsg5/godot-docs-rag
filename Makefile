# Godot Docs RAG - Makefile
# Simplified commands for Docker management

.PHONY: help setup run clean logs shell test status

# Default target
help:
	@echo "🔍 Godot Docs RAG - Available Commands"
	@echo "======================================"
	@echo ""
	@echo "  make setup     - Initial setup (creates .env, starts services)"
	@echo "  make run       - Run the complete RAG pipeline"
	@echo "  make clean     - Stop and remove all containers"
	@echo "  make logs      - Show logs from all services"
	@echo "  make shell     - Open bash shell in main container"
	@echo "  make status    - Show status of all services"
	@echo "  make test      - Test LLM connection"
	@echo ""
	@echo "Quick start: make setup && make run"

# Setup environment and services
setup:
	@echo "🚀 Setting up Godot Docs RAG..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "📝 Created .env file - please configure it"; fi
	docker-compose build
	docker-compose up -d ollama
	@echo "⏳ Waiting for Ollama to be ready..."
	@sleep 10
	docker-compose up ollama-init
	@echo "✅ Setup complete!"

# Run the pipeline
run:
	@echo "🔍 Running Godot Docs RAG pipeline..."
	docker-compose run --rm godot-docs-rag

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	@echo "✅ Cleanup complete"

# Show logs
logs:
	docker-compose logs -f

# Open shell
shell:
	docker-compose run --rm godot-docs-rag bash

# Show service status
status:
	docker-compose ps

# Test LLM connection
test:
	@echo "🧪 Testing LLM connection..."
	@if grep -q "LLM_PROVIDER=ollama" .env; then \
		curl -s http://localhost:11434/api/tags | jq '.models[].name' || echo "❌ Ollama not responding"; \
	else \
		echo "OpenAI provider configured - test will be done during pipeline run"; \
	fi

# Development shortcuts
dev-ollama:
	@echo "🦙 Starting Ollama only..."
	docker-compose up -d ollama

dev-logs-ollama:
	docker-compose logs -f ollama

dev-pull-model:
	docker-compose run --rm ollama-init
