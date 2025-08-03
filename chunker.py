from bs4 import BeautifulSoup
import html2text
import re
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Represents a processed document chunk with metadata"""
    content: str
    title: str
    section: str
    source_url: str
    chunk_type: str  # 'tutorial', 'reference', 'example', 'explanation'
    difficulty: str  # 'beginner', 'intermediate', 'advanced'
    keywords: List[str]

class EnhancedHTMLChunker:
    """Advanced HTML chunker optimized for Godot documentation Q&A"""
    
    def __init__(self):
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.body_width = 0  # Don't wrap lines
        
    def extract_metadata_from_path(self, file_path: str) -> Dict[str, str]:
        """Extract metadata from file path"""
        path_parts = Path(file_path).parts
        
        # Determine section from path
        if 'getting_started' in path_parts:
            section = 'Getting Started'
            difficulty = 'beginner'
        elif 'tutorials' in path_parts:
            section = 'Tutorials'
            difficulty = 'intermediate'
        elif 'classes' in path_parts:
            section = 'API Reference'
            difficulty = 'advanced'
        elif 'contributing' in path_parts:
            section = 'Contributing'
            difficulty = 'intermediate'
        else:
            section = 'General'
            difficulty = 'intermediate'
            
        return {'section': section, 'difficulty': difficulty}
    
    def clean_html_content(self, html_content: str) -> BeautifulSoup:
        """Clean and parse HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove navigation, headers, footers
        for tag in soup.find_all(['nav', 'header', 'footer', 'aside']):
            tag.decompose()
            
        # Remove script and style tags
        for tag in soup.find_all(['script', 'style']):
            tag.decompose()
            
        # Clean up code blocks - add context
        for code_block in soup.find_all(['pre', 'code']):
            # Find preceding explanation
            prev_text = ""
            prev_sibling = code_block.find_previous_sibling()
            if prev_sibling and prev_sibling.get_text().strip():
                prev_text = prev_sibling.get_text().strip()[-100:]  # Last 100 chars
                
            # Add context to code block
            if prev_text:
                code_block.insert(0, f"Context: {prev_text}\\n\\n")
                
        return soup
    
    def extract_meaningful_chunks(self, soup: BeautifulSoup, metadata: Dict) -> List[DocumentChunk]:
        """Extract semantically meaningful chunks from HTML"""
        chunks = []
        
        # Find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup
        
        # Extract title
        title_elem = main_content.find(['h1', 'title'])
        page_title = title_elem.get_text().strip() if title_elem else "Godot Documentation"
        
        # Process sections based on headers
        current_section = {"title": page_title, "content": [], "level": 0}
        sections = [current_section]
        
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'ul', 'ol', 'pre']):
            if element.name.startswith('h'):
                # New section
                level = int(element.name[1])
                section_title = element.get_text().strip()
                
                # Save previous section if it has content
                if current_section["content"]:
                    chunk_content = self._combine_content_elements(current_section["content"])
                    if len(chunk_content.strip()) > 50:  # Minimum content length
                        chunks.append(self._create_chunk(
                            chunk_content, 
                            current_section["title"],
                            metadata
                        ))
                
                # Start new section
                current_section = {"title": section_title, "content": [], "level": level}
                sections.append(current_section)
                
            else:
                # Add content to current section
                text = element.get_text().strip()
                if text and len(text) > 20:  # Filter out very short content
                    current_section["content"].append({
                        "type": element.name,
                        "text": text,
                        "is_code": element.name in ['pre', 'code']
                    })
        
        # Process final section
        if current_section["content"]:
            chunk_content = self._combine_content_elements(current_section["content"])
            if len(chunk_content.strip()) > 50:
                chunks.append(self._create_chunk(
                    chunk_content,
                    current_section["title"], 
                    metadata
                ))
        
        return chunks
    
    def _combine_content_elements(self, content_elements: List[Dict]) -> str:
        """Combine content elements into readable text"""
        combined = []
        
        for elem in content_elements:
            if elem["is_code"]:
                # Add context for code blocks
                combined.append(f"Code Example:\\n```\\n{elem['text']}\\n```")
            else:
                combined.append(elem["text"])
        
        return "\\n\\n".join(combined)
    
    def _create_chunk(self, content: str, title: str, metadata: Dict) -> DocumentChunk:
        """Create a DocumentChunk with metadata"""
        
        # Determine chunk type based on content
        chunk_type = self._classify_chunk_type(content)
        
        # Extract keywords
        keywords = self._extract_keywords(content, title)
        
        return DocumentChunk(
            content=content,
            title=title,
            section=metadata.get('section', 'General'),
            source_url=metadata.get('source_url', ''),
            chunk_type=chunk_type,
            difficulty=metadata.get('difficulty', 'intermediate'),
            keywords=keywords
        )
    
    def _classify_chunk_type(self, content: str) -> str:
        """Classify the type of content chunk"""
        content_lower = content.lower()
        
        if 'example' in content_lower or 'code example' in content_lower:
            return 'example'
        elif any(word in content_lower for word in ['how to', 'tutorial', 'step', 'first']):
            return 'tutorial'
        elif any(word in content_lower for word in ['class', 'method', 'property', 'signal']):
            return 'reference'
        else:
            return 'explanation'
    
    def _extract_keywords(self, content: str, title: str) -> List[str]:
        """Extract relevant keywords from content"""
        # Common Godot terms
        godot_terms = [
            'scene', 'node', 'script', 'signal', 'method', 'property',
            'animation', 'physics', 'collision', 'input', 'ui', 'gui',
            'shader', 'material', 'texture', 'mesh', 'camera', 'light',
            'audio', 'sound', 'music', 'export', 'import', 'project'
        ]
        
        keywords = []
        content_lower = content.lower()
        title_lower = title.lower()
        
        # Add title words as keywords
        title_words = re.findall(r'\\b\\w+\\b', title_lower)
        keywords.extend([word for word in title_words if len(word) > 3])
        
        # Add Godot-specific terms found in content
        for term in godot_terms:
            if term in content_lower:
                keywords.append(term)
        
        return list(set(keywords[:10]))  # Limit to 10 unique keywords
    
    def process_html_file(self, file_path: str) -> List[DocumentChunk]:
        """Process a complete HTML file into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Extract metadata from file path
            metadata = self.extract_metadata_from_path(file_path)
            metadata['source_url'] = file_path
            
            # Clean and parse HTML
            soup = self.clean_html_content(html_content)
            
            # Extract chunks
            chunks = self.extract_meaningful_chunks(soup, metadata)
            
            return chunks
            
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")
            return []
