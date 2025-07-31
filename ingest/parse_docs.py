"""
HTML Documentation Processor for Godot Docs
Since we're using pre-built HTML, we just need to organize and clean the files
"""

import os
import sys
import shutil
import logging
import time
from pathlib import Path
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class GodotHTMLProcessor:
    def __init__(self, source_dir="data/raw/godot-docs-html", output_dir="data/parsed"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.html_dir = self.output_dir / "html"
    
    def show_progress_bar(self, current, total, prefix="Progress", suffix="Complete", length=50):
        """Display a progress bar in the terminal"""
        if total == 0:
            return
        
        percent = (current / total) * 100
        filled_length = int(length * current // total)
        bar = '█' * filled_length + '░' * (length - filled_length)
        
        print(f'\r{prefix} |{bar}| {current}/{total} ({percent:.1f}%) {suffix}', end='', flush=True)
        
        if current == total:
            print()  # New line when complete
        
    def setup_directories(self):
        """Create necessary output directories"""
        self.html_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {self.html_dir}")
        
    def find_html_files(self):
        """Find all HTML files in the source directory"""
        html_files = list(self.source_dir.rglob("*.html"))
        logger.info(f"Found {len(html_files)} HTML files")
        return html_files
    
    def is_content_file(self, html_file: Path) -> bool:
        """Check if HTML file contains actual documentation content"""
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Skip navigation files, search pages, etc.
            skip_patterns = [
                'index.html',
                'search.html',
                'genindex.html',
                'modindex.html',
                '_static',
                '_sources'
            ]
            
            if any(pattern in str(html_file) for pattern in skip_patterns):
                logger.debug(f"Skipping navigation file: {html_file.name}")
                return False
            
            # Check if it has meaningful content
            content_div = soup.find('div', class_='document') or soup.find('main') or soup.find('article')
            if not content_div:
                logger.debug(f"No content div found in: {html_file.name}")
                return False
            
            # Check if it has enough text content (more than just navigation)
            text_content = content_div.get_text(strip=True)
            if len(text_content) < 500:  # Skip pages with minimal content
                logger.debug(f"Insufficient content ({len(text_content)} chars) in: {html_file.name}")
                return False
            
            logger.debug(f"Valid content file: {html_file.name} ({len(text_content)} chars)")
            return True
            
        except Exception as e:
            logger.warning(f"Error checking content for {html_file}: {e}")
            return False
    
    def clean_html_content(self, html_file: Path, output_file: Path):
        """Clean and copy HTML file to output directory"""
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Remove unnecessary elements
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            # Remove navigation and sidebar elements
            for class_name in ['sidebar', 'navigation', 'toctree-wrapper', 'sphinxsidebar']:
                for element in soup.find_all(class_=class_name):
                    element.decompose()
            
            # Save cleaned HTML
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(str(soup))
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning {html_file}: {e}")
            return False
    
    def process_files(self):
        """Process all HTML files"""
        print("\n🔍 Phase 1: Scanning for HTML files...")
        start_time = time.time()
        html_files = self.find_html_files()
        
        if not html_files:
            logger.error(f"No HTML files found in {self.source_dir}")
            return False
        
        print(f"✅ Found {len(html_files)} HTML files in {time.time() - start_time:.1f}s")
        
        print(f"\n📋 Phase 2: Filtering content files...")
        filter_start = time.time()
        processed_count = 0
        content_files = []
        
        # Filter with progress bar
        for i, html_file in enumerate(html_files):
            self.show_progress_bar(i + 1, len(html_files), 
                                 prefix="Filtering", 
                                 suffix=f"files ({len(content_files)} valid so far)")
            
            if self.is_content_file(html_file):
                content_files.append(html_file)
        
        filter_time = time.time() - filter_start
        print(f"✅ Filtered to {len(content_files)} content files in {filter_time:.1f}s")
        print(f"   📊 {len(content_files)}/{len(html_files)} files contain valid content ({(len(content_files)/len(html_files)*100):.1f}%)")
        
        if not content_files:
            logger.error("No valid content files found!")
            return False
        
        print(f"\n🔧 Phase 3: Processing and cleaning HTML files...")
        process_start = time.time()
        
        for i, html_file in enumerate(content_files):
            # Show detailed progress bar with ETA
            elapsed = time.time() - process_start
            if i > 0:
                eta = (elapsed / i) * (len(content_files) - i)
                eta_str = f"ETA: {eta:.1f}s"
            else:
                eta_str = "ETA: calculating..."
            
            self.show_progress_bar(i + 1, len(content_files), 
                                 prefix="Processing", 
                                 suffix=f"files | {eta_str}")
            
            # Create relative output path
            rel_path = html_file.relative_to(self.source_dir)
            output_file = self.html_dir / rel_path
            
            # Create output directory if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Clean and copy file
            if self.clean_html_content(html_file, output_file):
                processed_count += 1
        
        total_time = time.time() - start_time
        print(f"\n🎉 Successfully processed {processed_count}/{len(content_files)} HTML files")
        print(f"⏱️  Total processing time: {total_time:.1f}s")
        print(f"📁 Output location: {self.html_dir}")
        print(f"📊 Processing rate: {processed_count/total_time:.1f} files/second")
        return processed_count > 0
    
    def validate_source(self):
        """Validate that source directory exists and contains HTML files"""
        if not self.source_dir.exists():
            logger.error(f"Source directory does not exist: {self.source_dir}")
            return False
            
        html_files = list(self.source_dir.rglob("*.html"))
        if not html_files:
            logger.error(f"No HTML files found in {self.source_dir}")
            return False
            
        logger.info(f"Found {len(html_files)} HTML files to process")
        return True


def main():
    """Main function to run the processor"""
    print("🔍 Godot HTML Documentation Processor")
    print("=" * 50)
    print("📝 Processing HTML documentation files...")
    print("=" * 50)
    
    overall_start = time.time()
    processor = GodotHTMLProcessor()
    
    # Validate source
    print("🔎 Validating source directory...")
    if not processor.validate_source():
        print("❌ Source validation failed")
        sys.exit(1)
    
    # Setup directories
    print("📁 Setting up output directories...")
    processor.setup_directories()
    
    # Process files
    print("🚀 Starting HTML processing...")
    if processor.process_files():
        total_time = time.time() - overall_start
        print("\n" + "=" * 50)
        print("🎉 HTML processing completed successfully!")
        print(f"⏱️  Total execution time: {total_time:.1f} seconds")
        print("=" * 50)
    else:
        print("❌ HTML processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
