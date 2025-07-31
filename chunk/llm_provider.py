"""
LLM Provider abstraction for both OpenAI and Ollama
"""

import os
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the LLM"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the LLM provider is available"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider"""
    
    def __init__(self):
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.model = os.getenv("OPENAI_MODEL", "gpt-4")
            self.temperature = float(os.getenv("TEMPERATURE", "0.3"))
            self.max_tokens = int(os.getenv("MAX_TOKENS", "2000"))
            logger.info(f"OpenAI provider initialized with model: {self.model}")
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating educational Q&A pairs from technical documentation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return ""
    
    def test_connection(self) -> bool:
        """Test OpenAI API connection"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            logger.info("✅ OpenAI API connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ OpenAI API connection failed: {e}")
            return False


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider"""
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.temperature = float(os.getenv("TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "2000"))
        logger.info(f"Ollama provider initialized: {self.base_url} with model: {self.model}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                },
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return ""
    
    def test_connection(self) -> bool:
        """Test Ollama connection"""
        try:
            # Check if server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                logger.error("❌ Ollama server not responding")
                return False
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            if self.model not in model_names:
                logger.warning(f"⚠️ Model {self.model} not found. Available: {model_names}")
                # Try to pull the model
                logger.info(f"Attempting to pull model: {self.model}")
                pull_response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model},
                    timeout=600  # 10 minutes for model download
                )
                if pull_response.status_code != 200:
                    logger.error(f"❌ Failed to pull model {self.model}")
                    return False
            
            # Test generation
            test_response = self.generate("Hello", num_predict=5)
            if not test_response:
                logger.error("❌ Ollama test generation failed")
                return False
            
            logger.info("✅ Ollama connection successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {e}")
            logger.info("Make sure Ollama is running and accessible")
            return False


class LLMProvider:
    """Factory class for LLM providers"""
    
    def __new__(cls) -> BaseLLMProvider:
        """Create appropriate LLM provider based on configuration"""
        provider_type = os.getenv("LLM_PROVIDER", "ollama").lower()
        
        if provider_type == "openai":
            return OpenAIProvider()
        elif provider_type == "ollama":
            return OllamaProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {provider_type}. Use 'openai' or 'ollama'")
