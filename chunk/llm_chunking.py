"""
Legacy LLM chunking script - kept for backward compatibility
For the new Docker-based pipeline, see src/main.py and chunk/chunker.py
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import html2text

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalLLMChunker:
    def __init__(self, 
                 input_dir="data/parsed/html",
                 output_dir="data/chunks",
                 max_tokens=2000,
                 temperature=0.3):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Ollama configuration
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        
        logger.info(f"Using Ollama model: {self.ollama_model}")
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        # HTML to text converter
        self.h = html2text.HTML2Text()
        self.h.ignore_links = False
        self.h.ignore_images = True
        
    def _test_ollama_connection(self):
        """Test Ollama connection"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if self.ollama_model not in model_names:
                    logger.warning(f"Model {self.ollama_model} not found. Available models: {model_names}")
                    logger.info(f"Run: ollama pull {self.ollama_model}")
                else:
                    logger.info(f"✅ Ollama connected. Using model: {self.ollama_model}")
            else:
                logger.error("❌ Ollama server not responding. Make sure Ollama is running.")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            logger.info("Install Ollama from: https://ollama.ai")
    
    def setup_output_dir(self):
        """Create output directory if it doesn't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {self.output_dir}")
    
    def extract_text_from_html(self, html_content: str) -> str:
        """Convert HTML to clean markdown text"""
        try:
            # Use BeautifulSoup to clean up HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove navigation and footer elements
            for element in soup.find_all(['nav', 'footer', 'aside', 'script', 'style']):
                element.decompose()
            
            # Convert to markdown
            markdown_text = self.h.handle(str(soup))
            
            return markdown_text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ""
    
    def generate_with_ollama(self, prompt: str) -> str:
        """Generate response using Ollama"""
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                },
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            return ""
    
    def generate_qa_pairs(self, content: str, filename: str) -> List[Dict[str, Any]]:
        """Generate Q&A pairs from content using local LLM"""
        
        prompt = f"""You are an expert at creating educational Q&A pairs from technical documentation.

Analyze this Godot game engine documentation and generate comprehensive Q&A pairs that would be useful for developers learning Godot.

Guidelines:
1. Generate 3-10 Q&A pairs depending on content richness
2. Questions should be practical and actionable
3. Ensure answers are complete and include code examples when relevant
4. Cover different difficulty levels (beginner to advanced)
5. Focus on "how-to" questions that developers commonly ask
6. Return ONLY valid JSON array format

Output format (return ONLY the JSON array):
[
    {{
        "question": "How do I...",
        "answer": "To accomplish this, you...",
        "category": "topic category",
        "difficulty": "beginner|intermediate|advanced",
        "code_example": "code snippet if applicable or null",
        "source_file": "{filename}"
    }}
]

Analyze this Godot documentation content and generate Q&A pairs:

Filename: {filename}

Content:
{content[:3000]}

Generate practical Q&A pairs that would help Godot developers. Respond with JSON only:"""

        try:
            response_text = self.generate_with_ollama(prompt)
            
            if not response_text:
                logger.error(f"Empty response from LLM for {filename}")
                return []
            
            # Try to extract JSON from response
            try:
                # Look for JSON array in the response
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    qa_pairs = json.loads(json_str)
                    
                    # Ensure each pair has the source file
                    for pair in qa_pairs:
                        pair["source_file"] = filename
                        
                    return qa_pairs
                else:
                    logger.error(f"No valid JSON array found in response for {filename}")
                    return []
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for {filename}: {e}")
                # Try to create a basic Q&A pair from the response
                return [{
                    "question": f"What information is available in {filename}?",
                    "answer": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                    "category": "general",
                    "difficulty": "beginner",
                    "code_example": None,
                    "source_file": filename
                }]
                
        except Exception as e:
            logger.error(f"Error generating Q&A for {filename}: {e}")
            return []
    
    def process_file(self, html_file: Path) -> bool:
        """Process a single HTML file and generate Q&A pairs"""
        try:
            # Read HTML content
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Extract text content
            text_content = self.extract_text_from_html(html_content)
            
            if not text_content.strip():
                logger.warning(f"No text content extracted from {html_file}")
                return False
            
            # Generate Q&A pairs
            qa_pairs = self.generate_qa_pairs(text_content, html_file.name)
            
            if not qa_pairs:
                logger.warning(f"No Q&A pairs generated for {html_file}")
                return False
            
            # Save to JSON file
            output_file = self.output_dir / f"{html_file.stem}_qa.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Generated {len(qa_pairs)} Q&A pairs for {html_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {html_file}: {e}")
            return False
    
    def process_all_files(self):
        """Process all HTML files in the input directory"""
        html_files = list(self.input_dir.rglob("*.html"))
        
        if not html_files:
            logger.error(f"No HTML files found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(html_files)} HTML files to process")
        
        successful = 0
        total_qa_pairs = 0
        
        for html_file in html_files:
            logger.info(f"Processing: {html_file.name}")
            if self.process_file(html_file):
                successful += 1
                # Count Q&A pairs in the generated file
                output_file = self.output_dir / f"{html_file.stem}_qa.json"
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        pairs = json.load(f)
                        total_qa_pairs += len(pairs)
        
        logger.info(f"🎉 Processing complete!")
        logger.info(f"✅ Successfully processed: {successful}/{len(html_files)} files")
        logger.info(f"📝 Total Q&A pairs generated: {total_qa_pairs}")
        logger.info(f"📁 Output directory: {self.output_dir}")


def main():
    """Main function to run the chunking process"""
    try:
        chunker = LocalLLMChunker()
        chunker.setup_output_dir()
        chunker.process_all_files()
    except Exception as e:
        logger.error(f"❌ Chunking process failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
