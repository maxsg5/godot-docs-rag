"""
Document chunker for generating Q&A pairs from HTML documentation
Works with any LLM provider (OpenAI, Ollama, etc.)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup
import html2text

from legacy.chunk.llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Document chunker that generates Q&A pairs using LLM providers"""
    
    def __init__(self, 
                 llm_provider: BaseLLMProvider,
                 input_dir="data/parsed/html",
                 output_dir="data/chunks"):
        
        self.llm_provider = llm_provider
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # HTML to text converter with better settings for Godot docs
        self.h = html2text.HTML2Text()
        self.h.ignore_links = False
        self.h.ignore_images = True
        self.h.ignore_emphasis = False
        self.h.body_width = 0  # Don't wrap lines
        self.h.unicode_snob = True  # Better Unicode handling
        
        logger.info(f"DocumentChunker initialized with {type(llm_provider).__name__}")
    
    def setup_output_dir(self):
        """Create output directory if it doesn't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {self.output_dir}")
    
    def extract_text_from_html(self, html_content: str) -> str:
        """Convert HTML to clean markdown text optimized for Godot docs"""
        try:
            # Use BeautifulSoup to clean up HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unnecessary elements (already cleaned in processing, but double-check)
            for element in soup.find_all(['nav', 'footer', 'aside', 'script', 'style']):
                element.decompose()
            
            # Find the main content area
            main_content = (
                soup.find('div', class_='document') or 
                soup.find('main') or 
                soup.find('article') or
                soup.find('div', class_='body') or
                soup
            )
            
            # Convert to markdown
            markdown_text = self.h.handle(str(main_content))
            
            # Clean up the markdown text
            lines = markdown_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip empty lines and navigation artifacts
                if line and not line.startswith('¶') and len(line) > 1:
                    cleaned_lines.append(line)
            
            return '\n'.join(cleaned_lines)
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ""
    
    def create_qa_prompt(self, content: str, filename: str) -> str:
        """Create the prompt for Q&A generation"""
        return f"""You are an expert at creating educational Q&A pairs from technical documentation.

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
    
    def parse_qa_response(self, response_text: str, filename: str) -> List[Dict[str, Any]]:
        """Parse the LLM response to extract Q&A pairs"""
        if not response_text:
            logger.error(f"Empty response from LLM for {filename}")
            return []
        
        try:
            # Look for JSON array in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                qa_pairs = json.loads(json_str)
                
                # Ensure each pair has the source file
                for pair in qa_pairs:
                    if isinstance(pair, dict):
                        pair["source_file"] = filename
                
                return qa_pairs
            else:
                logger.error(f"No valid JSON array found in response for {filename}")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for {filename}: {e}")
            # Create a fallback Q&A pair
            return [{
                "question": f"What information is available in {filename}?",
                "answer": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                "category": "general",
                "difficulty": "beginner",
                "code_example": None,
                "source_file": filename
            }]
    
    def generate_qa_pairs(self, content: str, filename: str) -> List[Dict[str, Any]]:
        """Generate Q&A pairs from content using the LLM provider"""
        try:
            prompt = self.create_qa_prompt(content, filename)
            response_text = self.llm_provider.generate(prompt)
            return self.parse_qa_response(response_text, filename)
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
    
    def process_all_files(self) -> Tuple[int, int]:
        """Process all HTML files and return (success_count, total_qa_pairs)"""
        html_files = list(self.input_dir.rglob("*.html"))
        
        if not html_files:
            logger.error(f"No HTML files found in {self.input_dir}")
            return 0, 0
        
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
                    try:
                        with open(output_file, 'r') as f:
                            pairs = json.load(f)
                            total_qa_pairs += len(pairs)
                    except Exception as e:
                        logger.error(f"Error counting Q&A pairs in {output_file}: {e}")
        
        logger.info(f"🎉 Processing complete!")
        logger.info(f"✅ Successfully processed: {successful}/{len(html_files)} files")
        logger.info(f"📝 Total Q&A pairs generated: {total_qa_pairs}")
        
        return successful, total_qa_pairs
