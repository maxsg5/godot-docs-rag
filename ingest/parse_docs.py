"""
Godot Documentation Parser using Sphinx
Converts reStructuredText (.rst) files to HTML for further processing
"""

import os
import sys
from pathlib import Path
from sphinx.application import Sphinx
from sphinx.util.docutils import docutils_namespace
import tempfile
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GodotDocsParser:
    def __init__(self, source_dir="data/raw/godot-docs", output_dir="data/parsed"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.html_dir = self.output_dir / "html"
        self.doctree_dir = self.output_dir / "doctrees"
        
    def setup_directories(self):
        """Create necessary output directories"""
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.doctree_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directories: {self.html_dir}, {self.doctree_dir}")
        
    def build_docs(self):
        """Build documentation using Sphinx"""
        try:
            logger.info("Starting Sphinx build...")
            
            # Use docutils namespace to avoid conflicts
            with docutils_namespace():
                app = Sphinx(
                    srcdir=str(self.source_dir),
                    confdir=str(self.source_dir),
                    outdir=str(self.html_dir),
                    doctreedir=str(self.doctree_dir),
                    buildername="html",
                    confoverrides={
                        'extensions': ['sphinx.ext.autodoc', 'sphinx.ext.viewcode'],
                        'html_theme': 'basic',
                        'master_doc': 'index'
                    }
                )
                app.build()
                
            logger.info("✅ Sphinx build completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sphinx build failed: {str(e)}")
            return False
    
    def get_html_files(self):
        """Get list of generated HTML files"""
        html_files = list(self.html_dir.rglob("*.html"))
        logger.info(f"Found {len(html_files)} HTML files")
        return html_files
    
    def validate_source(self):
        """Validate that source directory exists and contains .rst files"""
        if not self.source_dir.exists():
            logger.error(f"Source directory does not exist: {self.source_dir}")
            return False
            
        rst_files = list(self.source_dir.rglob("*.rst"))
        if not rst_files:
            logger.error(f"No .rst files found in {self.source_dir}")
            return False
            
        logger.info(f"Found {len(rst_files)} .rst files to process")
        return True


def main():
    """Main function to run the parser"""
    parser = GodotDocsParser()
    
    # Validate source
    if not parser.validate_source():
        sys.exit(1)
    
    # Setup directories
    parser.setup_directories()
    
    # Build documentation
    if parser.build_docs():
        html_files = parser.get_html_files()
        logger.info(f"🎉 Successfully parsed {len(html_files)} documentation files!")
        logger.info(f"📁 Output location: {parser.html_dir}")
    else:
        logger.error("❌ Documentation parsing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
