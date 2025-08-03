import os
import yaml
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter, 
    HTMLSemanticPreservingSplitter,
    RecursiveCharacterTextSplitter
)
from tqdm import tqdm
from bs4 import Tag
import json

# Import our document loading functions
from LoadDataForLangChain import load_godot_docs


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("Creating default configuration...")
        create_default_config(config_path)
        return load_config(config_path)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return {}


def create_default_config(config_path: str):
    """Create a default configuration file."""
    default_config = {
        'document_loading': {
            'method': 'unstructured'
        },
        'text_splitting': {
            'method': 'html_header',
            'headers_to_split_on': [
                ['h1', 'Header 1'],
                ['h2', 'Header 2'],
                ['h3', 'Header 3'],
                ['h4', 'Header 4']
            ],
            'html_header': {
                'return_each_element': False
            },
            'html_section': {
                'xslt_path': None
            },
            'html_semantic': {
                'max_chunk_size': 2000,
                'chunk_overlap': 200,
                'separators': ['\n\n', '\n', '. ', '! ', '? '],
                'preserve_images': True,
                'preserve_videos': True,
                'elements_to_preserve': ['table', 'ul', 'ol', 'code', 'pre'],
                'denylist_tags': ['script', 'style', 'head', 'meta']
            }
        },
        'secondary_splitting': {
            'enabled': True,
            'chunk_size': 1000,
            'chunk_overlap': 100
        },
        'output': {
            'save_splits': True,
            'output_directory': 'data/processed',
            'save_metadata': True,
            'max_examples_to_show': 5
        }
    }
    
    with open(config_path, 'w') as file:
        yaml.dump(default_config, file, default_flow_style=False, indent=2)
    print(f"✅ Created default configuration file: {config_path}")


def custom_code_handler(element: Tag) -> str:
    """Custom handler for code elements."""
    data_lang = element.get("data-lang", "unknown")
    code_content = element.get_text()
    return f"<code:{data_lang}>{code_content}</code>"


def custom_pre_handler(element: Tag) -> str:
    """Custom handler for pre elements."""
    code_tag = element.find('code')
    if code_tag:
        data_lang = code_tag.get("data-lang", "unknown")
        code_content = code_tag.get_text()
        return f"<code:{data_lang}>{code_content}</code>"
    return f"<pre>{element.get_text()}</pre>"


def split_documents_html_header(documents: List[Document], config: Dict[str, Any]) -> List[Document]:
    """
    Split documents using HTMLHeaderTextSplitter.
    
    Args:
        documents (List[Document]): List of documents to split
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        List[Document]: List of split documents
    """
    print("🔀 Splitting documents with HTMLHeaderTextSplitter...")
    
    # Get configuration
    headers_config = config['text_splitting']['headers_to_split_on']
    headers_to_split_on = [(h[0], h[1]) for h in headers_config]
    return_each_element = config['text_splitting']['html_header']['return_each_element']
    
    # Create splitter
    splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_element=return_each_element
    )
    
    all_splits = []
    
    for doc in tqdm(documents, desc="Splitting with HTMLHeaderTextSplitter"):
        try:
            # Create HTML content from document
            html_content = f"<html><body>{doc.page_content}</body></html>"
            splits = splitter.split_text(html_content)
            
            # Preserve original metadata
            for split in splits:
                split.metadata.update(doc.metadata)
                
            all_splits.extend(splits)
        except Exception as e:
            print(f"❌ Error splitting document {doc.metadata.get('source', 'unknown')}: {e}")
            continue
    
    return all_splits


def split_documents_html_section(documents: List[Document], config: Dict[str, Any]) -> List[Document]:
    """
    Split documents using HTMLSectionSplitter.
    
    Args:
        documents (List[Document]): List of documents to split
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        List[Document]: List of split documents
    """
    print("🔀 Splitting documents with HTMLSectionSplitter...")
    
    # Get configuration
    headers_config = config['text_splitting']['headers_to_split_on']
    headers_to_split_on = [(h[0], h[1]) for h in headers_config]
    xslt_path = config['text_splitting']['html_section']['xslt_path']
    
    # Create splitter
    splitter_args = {'headers_to_split_on': headers_to_split_on}
    if xslt_path:
        splitter_args['xslt_path'] = xslt_path
        
    splitter = HTMLSectionSplitter(**splitter_args)
    
    all_splits = []
    
    for doc in tqdm(documents, desc="Splitting with HTMLSectionSplitter"):
        try:
            # Create HTML content from document
            html_content = f"<html><body>{doc.page_content}</body></html>"
            splits = splitter.split_text(html_content)
            
            # Preserve original metadata
            for split in splits:
                split.metadata.update(doc.metadata)
                
            all_splits.extend(splits)
        except Exception as e:
            print(f"❌ Error splitting document {doc.metadata.get('source', 'unknown')}: {e}")
            continue
    
    return all_splits


def split_documents_html_semantic(documents: List[Document], config: Dict[str, Any]) -> List[Document]:
    """
    Split documents using HTMLSemanticPreservingSplitter.
    
    Args:
        documents (List[Document]): List of documents to split
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        List[Document]: List of split documents
    """
    print("🔀 Splitting documents with HTMLSemanticPreservingSplitter...")
    
    # Get configuration
    headers_config = config['text_splitting']['headers_to_split_on']
    headers_to_split_on = [(h[0], h[1]) for h in headers_config]
    semantic_config = config['text_splitting']['html_semantic']
    
    # Create custom handlers
    custom_handlers = {
        'code': custom_code_handler,
        'pre': custom_pre_handler
    }
    
    # Create splitter
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=headers_to_split_on,
        max_chunk_size=semantic_config['max_chunk_size'],
        separators=semantic_config['separators'],
        preserve_images=semantic_config['preserve_images'],
        preserve_videos=semantic_config['preserve_videos'],
        elements_to_preserve=semantic_config['elements_to_preserve'],
        denylist_tags=semantic_config['denylist_tags'],
        custom_handlers=custom_handlers
    )
    
    all_splits = []
    
    for doc in tqdm(documents, desc="Splitting with HTMLSemanticPreservingSplitter"):
        try:
            # Create HTML content from document
            html_content = f"<html><body>{doc.page_content}</body></html>"
            splits = splitter.split_text(html_content)
            
            # Preserve original metadata
            for split in splits:
                split.metadata.update(doc.metadata)
                
            all_splits.extend(splits)
        except Exception as e:
            print(f"❌ Error splitting document {doc.metadata.get('source', 'unknown')}: {e}")
            continue
    
    return all_splits


def apply_secondary_splitting(documents: List[Document], config: Dict[str, Any]) -> List[Document]:
    """
    Apply secondary character-based splitting to constrain chunk sizes.
    
    Args:
        documents (List[Document]): List of documents to split further
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        List[Document]: List of further split documents
    """
    if not config['secondary_splitting']['enabled']:
        return documents
    
    print("✂️ Applying secondary character-based splitting...")
    
    chunk_size = config['secondary_splitting']['chunk_size']
    chunk_overlap = config['secondary_splitting']['chunk_overlap']
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"📊 Secondary splitting created {len(splits)} chunks from {len(documents)} documents")
    
    return splits


def analyze_splits(splits: List[Document], config: Dict[str, Any]):
    """
    Analyze and display information about the split documents.
    
    Args:
        splits (List[Document]): List of split documents
        config (Dict[str, Any]): Configuration dictionary
    """
    if not splits:
        print("❌ No splits to analyze")
        return
    
    print(f"\n📊 Split Analysis:")
    print(f"Total splits: {len(splits)}")
    
    # Calculate statistics
    content_lengths = [len(split.page_content) for split in splits]
    avg_length = sum(content_lengths) / len(content_lengths)
    max_length = max(content_lengths)
    min_length = min(content_lengths)
    
    print(f"Average chunk length: {avg_length:.0f} characters")
    print(f"Longest chunk: {max_length} characters")
    print(f"Shortest chunk: {min_length} characters")
    
    # Show metadata keys
    if splits:
        metadata_keys = set()
        for split in splits[:10]:  # Sample first 10 for metadata analysis
            metadata_keys.update(split.metadata.keys())
        print(f"Metadata keys found: {list(metadata_keys)}")
    
    # Show examples
    max_examples = config['output']['max_examples_to_show']
    print(f"\n📝 Example splits (showing first {max_examples}):")
    for i, split in enumerate(splits[:max_examples]):
        print(f"\n--- Split {i+1} ---")
        print(f"Source: {split.metadata.get('source', 'Unknown')}")
        for key, value in split.metadata.items():
            if key != 'source':
                print(f"{key}: {value}")
        print(f"Content length: {len(split.page_content)} characters")
        print(f"Content preview: {split.page_content[:200]}...")


def save_splits(splits: List[Document], config: Dict[str, Any]):
    """
    Save the split documents to files.
    
    Args:
        splits (List[Document]): List of split documents
        config (Dict[str, Any]): Configuration dictionary
    """
    if not config['output']['save_splits']:
        return
    
    output_dir = Path(config['output']['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving splits to: {output_dir}")
    
    # Save as pickle for easy loading
    pickle_path = output_dir / "godot_docs_splits.pickle"
    with open(pickle_path, 'wb') as f:
        pickle.dump(splits, f)
    print(f"✅ Saved {len(splits)} splits to {pickle_path}")
    
    # Save metadata if requested
    if config['output']['save_metadata']:
        metadata_path = output_dir / "splits_metadata.json"
        metadata = []
        for i, split in enumerate(splits):
            metadata.append({
                'chunk_id': i,
                'content_length': len(split.page_content),
                'metadata': split.metadata
            })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved metadata to {metadata_path}")
    
    # Save a sample as text for inspection
    sample_path = output_dir / "sample_splits.txt"
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write(f"Godot Documentation Splits Sample\n")
        f.write(f"Generated from {len(splits)} total splits\n")
        f.write("=" * 50 + "\n\n")
        
        for i, split in enumerate(splits[:10]):  # Save first 10 as sample
            f.write(f"Split {i+1}:\n")
            f.write(f"Source: {split.metadata.get('source', 'Unknown')}\n")
            for key, value in split.metadata.items():
                if key != 'source':
                    f.write(f"{key}: {value}\n")
            f.write(f"Content length: {len(split.page_content)} characters\n")
            f.write("-" * 30 + "\n")
            f.write(f"{split.page_content}\n")
            f.write("\n" + "=" * 50 + "\n\n")
    
    print(f"✅ Saved sample splits to {sample_path}")


def main():
    """Main function to run the text splitting pipeline."""
    print("🚀 Godot Documentation Text Splitting Pipeline")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    if not config:
        print("❌ Failed to load configuration")
        return
    
    print(f"📋 Configuration loaded:")
    print(f"  - Document loading method: {config['document_loading']['method']}")
    print(f"  - Text splitting method: {config['text_splitting']['method']}")
    print(f"  - Secondary splitting: {'enabled' if config['secondary_splitting']['enabled'] else 'disabled'}")
    
    # Load documents
    print("\n📚 Loading documents...")
    documents = load_godot_docs(method=config['document_loading']['method'])
    
    if not documents:
        print("❌ No documents loaded. Please run LoadDataForLangChain.py first.")
        return
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Split documents based on method
    splitting_method = config['text_splitting']['method']
    
    if splitting_method == 'html_header':
        splits = split_documents_html_header(documents, config)
    elif splitting_method == 'html_section':
        splits = split_documents_html_section(documents, config)
    elif splitting_method == 'html_semantic':
        splits = split_documents_html_semantic(documents, config)
    else:
        print(f"❌ Unknown splitting method: {splitting_method}")
        return
    
    if not splits:
        print("❌ No splits created")
        return
    
    print(f"✅ Created {len(splits)} initial splits")
    
    # Apply secondary splitting if enabled
    final_splits = apply_secondary_splitting(splits, config)
    
    # Analyze results
    analyze_splits(final_splits, config)
    
    # Save results
    save_splits(final_splits, config)
    
    print(f"\n🎉 Text splitting pipeline completed!")
    print(f"📊 Final result: {len(final_splits)} text chunks ready for RAG")


if __name__ == "__main__":
    main()
