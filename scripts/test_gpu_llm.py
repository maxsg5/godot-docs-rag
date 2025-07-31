#!/usr/bin/env python3
"""
GPU and Local LLM Test Script
Tests different local LLM setups and GPU availability
"""

import sys
import subprocess
import importlib.util
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check GPU availability and CUDA setup"""
    logger.info("🔍 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            logger.info(f"✅ CUDA available: {cuda_version}")
            logger.info(f"✅ GPU count: {gpu_count}")
            logger.info(f"✅ GPU 0: {gpu_name}")
            
            # Test GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ GPU memory: {gpu_memory:.1f} GB")
            
            return True
        else:
            logger.warning("⚠️  CUDA not available - will use CPU")
            return False
    except ImportError:
        logger.warning("⚠️  PyTorch not installed - cannot check CUDA")
        return False

def check_ollama_setup():
    """Check if Ollama is installed and running"""
    logger.info("🦙 Checking Ollama setup...")
    
    try:
        import requests
        
        # Check if Ollama server is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info(f"✅ Ollama server running with {len(models)} models:")
            for model in models[:3]:  # Show first 3 models
                logger.info(f"   - {model['name']}")
            return True
        else:
            logger.warning("⚠️  Ollama server not responding")
            return False
            
    except requests.exceptions.RequestException:
        logger.warning("⚠️  Ollama server not running")
        logger.info("💡 Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        logger.info("💡 Start server: ollama serve")
        logger.info("💡 Pull model: ollama pull llama3.1:8b")
        return False
    except ImportError:
        logger.warning("⚠️  requests library not available")
        return False

def test_ollama_inference():
    """Test Ollama inference with a simple prompt"""
    logger.info("🧪 Testing Ollama inference...")
    
    try:
        import requests
        
        payload = {
            "model": "llama3.1:8b",
            "prompt": "What is Godot game engine? Answer in one sentence.",
            "options": {
                "temperature": 0.3,
                "num_predict": 50
            },
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()["response"]
            logger.info(f"✅ Ollama inference successful!")
            logger.info(f"📝 Response: {result[:100]}...")
            return True
        else:
            logger.error(f"❌ Ollama inference failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ollama inference error: {e}")
        return False

def check_transformers_setup():
    """Check HuggingFace Transformers setup"""
    logger.info("🤗 Checking HuggingFace Transformers setup...")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Try loading a small model
        model_name = "microsoft/DialoGPT-small"  # Smaller model for testing
        logger.info(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        if torch.cuda.is_available():
            model = model.to("cuda")
            logger.info("✅ Model loaded on GPU")
        else:
            logger.info("✅ Model loaded on CPU")
            
        # Test inference
        prompt = "Godot is a"
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=20, do_sample=True, temperature=0.7)
            
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Transformers inference successful!")
        logger.info(f"📝 Response: {result}")
        
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️  Transformers not available: {e}")
        logger.info("💡 Install: pip install torch transformers accelerate")
        return False
    except Exception as e:
        logger.error(f"❌ Transformers error: {e}")
        return False

def run_system_info():
    """Display system information"""
    logger.info("💻 System Information:")
    
    try:
        # Python version
        logger.info(f"Python: {sys.version.split()[0]}")
        
        # GPU info via nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')[0]
            logger.info(f"GPU: {gpu_info}")
        else:
            logger.info("GPU: nvidia-smi not available")
            
    except Exception as e:
        logger.info(f"Could not get system info: {e}")

def main():
    """Run all tests"""
    print("🚀 Local LLM GPU Test Suite")
    print("="*50)
    
    run_system_info()
    print()
    
    # Test GPU availability
    gpu_available = check_gpu_availability()
    print()
    
    # Test Ollama (recommended approach)
    ollama_works = check_ollama_setup()
    if ollama_works:
        test_ollama_inference()
    print()
    
    # Test Transformers (alternative approach)
    transformers_works = check_transformers_setup()
    print()
    
    # Summary
    print("📋 Test Summary:")
    print(f"GPU Available: {'✅' if gpu_available else '❌'}")
    print(f"Ollama Ready: {'✅' if ollama_works else '❌'}")
    print(f"Transformers Ready: {'✅' if transformers_works else '❌'}")
    print()
    
    if ollama_works:
        print("🎉 Recommended: Use Ollama for local LLM inference")
        print("   Set LLM_PROVIDER=ollama in your .env file")
    elif transformers_works:
        print("🎉 Alternative: Use HuggingFace Transformers")
        print("   Set LLM_PROVIDER=transformers in your .env file")
    else:
        print("❌ No local LLM setup is working")
        print("💡 Try installing Ollama first: https://ollama.ai")

if __name__ == "__main__":
    main()
