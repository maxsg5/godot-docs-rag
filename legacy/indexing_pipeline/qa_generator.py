"""
Q&A Generation for Godot Documentation
=====================================

This module generates question-answer pairs from the chunked Godot documentation
for training and evaluation of the RAG system.

The Q&A pairs will be used to:
1. Test retrieval quality
2. Evaluate answer generation
3. Create a benchmark dataset
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from tqdm import tqdm

# LangChain imports
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """Represents a question-answer pair with metadata"""
    question: str
    answer: str
    source_chunk_id: str
    source_file: str
    headers: Dict[str, str]
    difficulty: str  # "basic", "intermediate", "advanced"
    category: str    # "getting_started", "scripting", "rendering", etc.
    context: str     # The original chunk content


class QAGenerator:
    """Generates Q&A pairs from document chunks"""
    
    def __init__(self, vector_store_path: str = "output/chroma_db"):
        """
        Initialize Q&A generator
        
        Args:
            vector_store_path: Path to the ChromaDB vector store
        """
        self.vector_store_path = Path(vector_store_path)
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load vector store
        if self.vector_store_path.exists():
            self.vector_store = Chroma(
                persist_directory=str(self.vector_store_path),
                embedding_function=self.embeddings,
                collection_name="godot_docs"
            )
        else:
            raise FileNotFoundError(f"Vector store not found at {vector_store_path}")
    
    def categorize_content(self, chunk: Document) -> str:
        """Categorize content based on source path and headers"""
        source = chunk.metadata.get("source", "").lower()
        headers = [v.lower() for k, v in chunk.metadata.items() if k.startswith("Header")]
        content = chunk.page_content.lower()
        
        # Category mapping based on common Godot documentation structure
        if any(term in source for term in ["getting_started", "step_by_step", "introduction"]):
            return "getting_started"
        elif any(term in source for term in ["scripting", "gdscript", "programming"]):
            return "scripting"
        elif any(term in source for term in ["rendering", "visual", "shader", "material"]):
            return "rendering"
        elif any(term in source for term in ["physics", "collision", "rigidbody"]):
            return "physics"
        elif any(term in source for term in ["audio", "sound", "music"]):
            return "audio"
        elif any(term in source for term in ["input", "control", "ui", "gui"]):
            return "ui_input"
        elif any(term in source for term in ["networking", "multiplayer"]):
            return "networking"
        elif any(term in source for term in ["export", "platform", "deploy"]):
            return "deployment"
        elif any(term in source for term in ["plugin", "addon", "tool"]):
            return "plugins"
        else:
            return "general"
    
    def determine_difficulty(self, chunk: Document) -> str:
        """Determine difficulty level based on content complexity"""
        content = chunk.page_content.lower()
        headers = [v.lower() for k, v in chunk.metadata.items() if k.startswith("Header")]
        
        # Basic indicators
        basic_terms = [
            "introduction", "getting started", "first", "basic", "simple", 
            "what is", "overview", "create your first"
        ]
        
        # Advanced indicators  
        advanced_terms = [
            "advanced", "optimization", "performance", "custom", "extension",
            "plugin development", "engine internals", "native", "gdextension"
        ]
        
        # Check headers first
        header_text = " ".join(headers)
        if any(term in header_text for term in basic_terms):
            return "basic"
        elif any(term in header_text for term in advanced_terms):
            return "advanced"
        
        # Check content
        if any(term in content for term in basic_terms):
            return "basic"
        elif any(term in content for term in advanced_terms):
            return "advanced"
        else:
            return "intermediate"
    
    def generate_questions_from_chunk(self, chunk: Document) -> List[QAPair]:
        """Generate multiple Q&A pairs from a document chunk"""
        qa_pairs = []
        
        # Extract key information
        source = chunk.metadata.get("source", "unknown")
        chunk_id = chunk.metadata.get("chunk_id", "unknown")
        headers = {k: v for k, v in chunk.metadata.items() if k.startswith("Header")}
        category = self.categorize_content(chunk)
        difficulty = self.determine_difficulty(chunk)
        content = chunk.page_content
        
        # Generate different types of questions based on content
        questions = self._generate_question_templates(chunk, headers, category, difficulty)
        
        for question_data in questions:
            qa_pair = QAPair(
                question=question_data["question"],
                answer=question_data["answer"],
                source_chunk_id=chunk_id,
                source_file=source,
                headers=headers,
                difficulty=difficulty,
                category=category,
                context=content
            )
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def _generate_question_templates(self, chunk: Document, headers: Dict, category: str, difficulty: str) -> List[Dict]:
        """Generate question templates based on content analysis"""
        content = chunk.page_content
        questions = []
        
        # Get the main topic from headers
        main_topic = headers.get("Header 1", headers.get("Header 2", "Godot"))
        
        # Template 1: Direct factual questions
        if len(content) > 100:
            questions.append({
                "question": f"What is {main_topic} in Godot?",
                "answer": self._extract_definition_from_content(content, main_topic)
            })
        
        # Template 2: How-to questions
        if "how to" in content.lower() or any(word in content.lower() for word in ["create", "use", "implement", "setup"]):
            questions.append({
                "question": f"How do you use {main_topic} in Godot?",
                "answer": self._extract_howto_from_content(content, main_topic)
            })
        
        # Template 3: Feature-specific questions
        if category == "scripting":
            questions.append({
                "question": f"How do you implement {main_topic} in GDScript?",
                "answer": self._extract_code_examples(content)
            })
        
        # Template 4: Troubleshooting questions
        if any(word in content.lower() for word in ["error", "issue", "problem", "troubleshoot", "warning"]):
            questions.append({
                "question": f"How do you troubleshoot issues with {main_topic} in Godot?",
                "answer": self._extract_troubleshooting_info(content)
            })
        
        # Template 5: Comparison questions
        if "vs" in content.lower() or "compared to" in content.lower():
            questions.append({
                "question": f"How does {main_topic} compare to other approaches in Godot?",
                "answer": content[:500] + "..." if len(content) > 500 else content
            })
        
        return questions
    
    def _extract_definition_from_content(self, content: str, topic: str) -> str:
        """Extract definition or explanation from content"""
        sentences = content.split('. ')
        
        # Look for sentences that define or explain the topic
        for sentence in sentences[:3]:  # Check first 3 sentences
            if any(word in sentence.lower() for word in [topic.lower(), "is", "defines", "refers to"]):
                return sentence.strip() + '.'
        
        # Fallback to first paragraph
        paragraphs = content.split('\n\n')
        return paragraphs[0][:300] + "..." if len(paragraphs[0]) > 300 else paragraphs[0]
    
    def _extract_howto_from_content(self, content: str, topic: str) -> str:
        """Extract how-to information from content"""
        # Look for step-by-step instructions or procedural content
        lines = content.split('\n')
        
        howto_lines = []
        for line in lines:
            if any(indicator in line.lower() for indicator in [
                "step", "first", "then", "next", "finally", "to create", "to use", "to implement"
            ]):
                howto_lines.append(line.strip())
        
        if howto_lines:
            return '\n'.join(howto_lines[:5])  # First 5 relevant lines
        
        # Fallback to first paragraph that seems instructional
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if any(word in para.lower() for word in ["create", "use", "implement", "setup", "configure"]):
                return para[:400] + "..." if len(para) > 400 else para
        
        return content[:300] + "..." if len(content) > 300 else content
    
    def _extract_code_examples(self, content: str) -> str:
        """Extract code examples and explanations"""
        # Look for code blocks or code-related content
        code_sections = []
        
        # Simple detection of code-like content
        lines = content.split('\n')
        in_code_block = False
        current_block = []
        
        for line in lines:
            # Detect code blocks (basic heuristic)
            if any(indicator in line for indicator in ['```', 'func ', 'var ', 'extends', 'class_name']):
                if not in_code_block:
                    in_code_block = True
                    current_block = [line]
                else:
                    current_block.append(line)
                    if '```' in line:
                        code_sections.append('\n'.join(current_block))
                        current_block = []
                        in_code_block = False
            elif in_code_block:
                current_block.append(line)
        
        if code_sections:
            return '\n\n'.join(code_sections[:2])  # First 2 code blocks
        
        # Fallback to content that mentions code-related terms
        return content[:400] + "..." if len(content) > 400 else content
    
    def _extract_troubleshooting_info(self, content: str) -> str:
        """Extract troubleshooting information"""
        troubleshooting_keywords = [
            "error", "issue", "problem", "fix", "solve", "troubleshoot", 
            "warning", "debug", "common mistakes"
        ]
        
        sentences = content.split('. ')
        relevant_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in troubleshooting_keywords):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return '. '.join(relevant_sentences[:3]) + '.'
        
        return content[:300] + "..." if len(content) > 300 else content
    
    def generate_qa_dataset(self, max_chunks: Optional[int] = None) -> List[QAPair]:
        """Generate Q&A dataset from all chunks in the vector store"""
        logger.info("🤖 Generating Q&A pairs from document chunks...")
        
        # Get all documents from vector store
        collection = self.vector_store._collection
        all_docs = collection.get()
        
        if max_chunks:
            # Limit the number of chunks to process
            doc_ids = all_docs['ids'][:max_chunks]
            metadatas = all_docs['metadatas'][:max_chunks]
            documents = all_docs['documents'][:max_chunks]
        else:
            doc_ids = all_docs['ids']
            metadatas = all_docs['metadatas']
            documents = all_docs['documents']
        
        all_qa_pairs = []
        
        for doc_id, metadata, content in tqdm(
            zip(doc_ids, metadatas, documents), 
            total=len(doc_ids), 
            desc="Generating Q&A pairs"
        ):
            # Create Document object
            doc = Document(page_content=content, metadata=metadata)
            
            # Generate Q&A pairs for this chunk
            qa_pairs = self.generate_questions_from_chunk(doc)
            all_qa_pairs.extend(qa_pairs)
        
        logger.info(f"✅ Generated {len(all_qa_pairs)} Q&A pairs from {len(doc_ids)} chunks")
        return all_qa_pairs
    
    def save_qa_dataset(self, qa_pairs: List[QAPair], output_file: str = "output/godot_qa_dataset.json"):
        """Save Q&A dataset to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        qa_data = {
            "metadata": {
                "total_pairs": len(qa_pairs),
                "categories": list(set(qa.category for qa in qa_pairs)),
                "difficulties": list(set(qa.difficulty for qa in qa_pairs)),
                "source_files": list(set(qa.source_file for qa in qa_pairs))
            },
            "qa_pairs": [
                {
                    "id": f"qa_{i:05d}",
                    "question": qa.question,
                    "answer": qa.answer,
                    "source_chunk_id": qa.source_chunk_id,
                    "source_file": qa.source_file,
                    "headers": qa.headers,
                    "difficulty": qa.difficulty,
                    "category": qa.category,
                    "context": qa.context[:500] + "..." if len(qa.context) > 500 else qa.context
                }
                for i, qa in enumerate(qa_pairs)
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Q&A dataset saved to {output_path}")
        
        # Generate statistics
        self._generate_statistics(qa_pairs, output_path.parent / "qa_statistics.json")
    
    def _generate_statistics(self, qa_pairs: List[QAPair], stats_file: Path):
        """Generate statistics about the Q&A dataset"""
        from collections import Counter
        
        stats = {
            "total_pairs": len(qa_pairs),
            "categories": dict(Counter(qa.category for qa in qa_pairs)),
            "difficulties": dict(Counter(qa.difficulty for qa in qa_pairs)),
            "source_distribution": dict(Counter(qa.source_file for qa in qa_pairs)),
            "average_question_length": sum(len(qa.question) for qa in qa_pairs) / len(qa_pairs),
            "average_answer_length": sum(len(qa.answer) for qa in qa_pairs) / len(qa_pairs),
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Statistics saved to {stats_file}")


def main():
    """Main entry point for Q&A generation"""
    logger.info("🚀 Starting Q&A Dataset Generation")
    
    # Initialize generator
    qa_generator = QAGenerator("output/chroma_db")
    
    # Generate Q&A pairs (limit to 1000 chunks for testing)
    qa_pairs = qa_generator.generate_qa_dataset(max_chunks=1000)
    
    # Save dataset
    qa_generator.save_qa_dataset(qa_pairs)
    
    logger.info("🎉 Q&A generation completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
