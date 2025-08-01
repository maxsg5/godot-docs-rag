import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
from langchain.schema import Document
from tqdm import tqdm
import glob


def find_html_files(directory: str) -> List[str]:
    """
    Find all HTML files in the given directory and subdirectories.
    
    Args:
        directory (str): The directory to search for HTML files
        
    Returns:
        List[str]: List of paths to HTML files
    """
    html_files = []
    
    # Use glob to find all HTML files recursively
    pattern = os.path.join(directory, "**", "*.html")
    html_files = glob.glob(pattern, recursive=True)
    
    return html_files


def load_html_with_unstructured(file_paths: List[str]) -> List[Document]:
    """
    Load HTML files using UnstructuredHTMLLoader.
    
    Args:
        file_paths (List[str]): List of HTML file paths
        
    Returns:
        List[Document]: List of LangChain Document objects
    """
    documents = []
    
    print("📄 Loading HTML files with Unstructured...")
    
    for file_path in tqdm(file_paths, desc="Processing with Unstructured"):
        try:
            loader = UnstructuredHTMLLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"❌ Error loading {file_path} with Unstructured: {e}")
            continue
    
    return documents


def load_html_with_bs4(file_paths: List[str]) -> List[Document]:
    """
    Load HTML files using BSHTMLLoader (BeautifulSoup4).
    
    Args:
        file_paths (List[str]): List of HTML file paths
        
    Returns:
        List[Document]: List of LangChain Document objects
    """
    documents = []
    
    print("🍲 Loading HTML files with BeautifulSoup4...")
    
    for file_path in tqdm(file_paths, desc="Processing with BS4"):
        try:
            loader = BSHTMLLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"❌ Error loading {file_path} with BS4: {e}")
            continue
    
    return documents


def load_godot_docs(method: str = "unstructured") -> List[Document]:
    """
    Load all Godot documentation HTML files into LangChain Documents.
    
    Args:
        method (str): Loading method - "unstructured" or "bs4"
        
    Returns:
        List[Document]: List of loaded documents
    """
    # Path to the extracted Godot documentation
    docs_dir = "data/raw"
    
    if not os.path.exists(docs_dir):
        print(f"❌ Documentation directory not found: {docs_dir}")
        print("Please run LoadData.py first to download the documentation.")
        return []
    
    print(f"🔍 Searching for HTML files in: {docs_dir}")
    html_files = find_html_files(docs_dir)
    print(f"📊 Found {len(html_files)} HTML files")
    
    if not html_files:
        print("❌ No HTML files found!")
        return []
    
    # Load documents based on selected method
    if method.lower() == "unstructured":
        documents = load_html_with_unstructured(html_files)
    elif method.lower() == "bs4":
        documents = load_html_with_bs4(html_files)
    else:
        print(f"❌ Unknown method: {method}")
        print("Available methods: 'unstructured', 'bs4'")
        return []
    
    print(f"✅ Successfully loaded {len(documents)} documents")
    return documents


def analyze_documents(documents: List[Document], max_examples: int = 3):
    """
    Analyze and display information about the loaded documents.
    
    Args:
        documents (List[Document]): List of documents to analyze
        max_examples (int): Maximum number of example documents to display
    """
    if not documents:
        print("❌ No documents to analyze")
        return
    
    print(f"\n📊 Document Analysis:")
    print(f"Total documents: {len(documents)}")
    
    # Calculate statistics
    content_lengths = [len(doc.page_content) for doc in documents]
    avg_length = sum(content_lengths) / len(content_lengths)
    max_length = max(content_lengths)
    min_length = min(content_lengths)
    
    print(f"Average content length: {avg_length:.0f} characters")
    print(f"Longest document: {max_length} characters")
    print(f"Shortest document: {min_length} characters")
    
    # Show examples
    print(f"\n📝 Example documents (showing first {max_examples}):")
    for i, doc in enumerate(documents[:max_examples]):
        print(f"\n--- Document {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        if 'title' in doc.metadata:
            print(f"Title: {doc.metadata['title']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:200]}...")


def save_documents_summary(documents: List[Document], output_file: str = "godot_docs_summary.txt"):
    """
    Save a summary of all documents to a text file.
    
    Args:
        documents (List[Document]): List of documents
        output_file (str): Output file path
    """
    print(f"💾 Saving documents summary to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Godot Documentation Summary\n")
        f.write(f"Generated from {len(documents)} HTML documents\n")
        f.write("=" * 50 + "\n\n")
        
        for i, doc in enumerate(documents):
            f.write(f"Document {i+1}:\n")
            f.write(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
            if 'title' in doc.metadata:
                f.write(f"Title: {doc.metadata['title']}\n")
            f.write(f"Content length: {len(doc.page_content)} characters\n")
            f.write("-" * 30 + "\n")
            f.write(f"{doc.page_content}\n")
            f.write("\n" + "=" * 50 + "\n\n")
    
    print(f"✅ Summary saved to: {output_file}")


if __name__ == "__main__":
    print("🚀 Loading Godot Documentation with LangChain")
    print("=" * 50)
    
    # Test both methods with a smaller sample for demonstration
    print("\n🧪 Testing both loading methods...")
    
    # Method 1: Unstructured (default)
    print("\n1️⃣ Testing Unstructured method:")
    method = "unstructured"
    print(f"📚 Using loading method: {method}")
    documents_unstructured = load_godot_docs(method=method)
    
    if documents_unstructured:
        print(f"✅ Unstructured loaded {len(documents_unstructured)} documents")
        
        # Show a quick comparison
        sample_doc = documents_unstructured[0] if documents_unstructured else None
        if sample_doc:
            print(f"📄 Sample document metadata: {list(sample_doc.metadata.keys())}")
    
    # Optional: Test BeautifulSoup4 method (commented out for faster execution)
    # Uncomment the lines below to test BS4 method
    """
    print("\n2️⃣ Testing BeautifulSoup4 method:")
    method = "bs4"
    print(f"📚 Using loading method: {method}")
    documents_bs4 = load_godot_docs(method=method)
    
    if documents_bs4:
        print(f"✅ BS4 loaded {len(documents_bs4)} documents")
        
        # Show comparison
        sample_doc = documents_bs4[0] if documents_bs4 else None
        if sample_doc:
            print(f"📄 Sample document metadata: {list(sample_doc.metadata.keys())}")
    """
    
    # Analyze the loaded documents
    if documents_unstructured:
        analyze_documents(documents_unstructured)
        
        # Optionally save summary (uncomment if needed)
        # save_documents_summary(documents_unstructured)
        
        print(f"\n🎉 Successfully processed Godot documentation!")
        print(f"📊 Total documents loaded: {len(documents_unstructured)}")
        print(f"🔍 Ready for RAG implementation!")
    else:
        print("❌ Failed to load any documents")
