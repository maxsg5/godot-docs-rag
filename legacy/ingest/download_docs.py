"""
Download pre-built Godot HTML documentation from nightly builds
Much faster and more reliable than parsing reStructuredText
"""

import os
import logging
import requests
import zipfile
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def download_godot_html_docs(
    target_dir="data/raw/godot-docs-html",
    version="stable"
) -> bool:
    """
    Download pre-built Godot HTML documentation
    
    Args:
        target_dir: Directory to extract the docs into
        version: Version to download ('stable', 'latest', '3.6')
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # URLs for different versions
    urls = {
        "stable": "https://nightly.link/godotengine/godot-docs/workflows/build_offline_docs/master/godot-docs-html-stable.zip",
        "latest": "https://nightly.link/godotengine/godot-docs/workflows/build_offline_docs/master/godot-docs-html-latest.zip", 
        "3.6": "https://nightly.link/godotengine/godot-docs/workflows/build_offline_docs/3.6/godot-docs-html-stable.zip"
    }
    
    if version not in urls:
        logger.error(f"Unknown version: {version}. Available: {list(urls.keys())}")
        return False
    
    url = urls[version]
    target_path = Path(target_dir)
    zip_path = target_path.parent / f"godot-docs-{version}.zip"
    
    try:
        # Create target directory
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if docs already exist
        if target_path.exists() and any(target_path.iterdir()):
            logger.info(f"Godot {version} docs already exist at {target_path}")
            return True
        
        logger.info(f"Downloading Godot {version} HTML documentation...")
        logger.info(f"URL: {url}")
        
        # Download the ZIP file
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress for large downloads
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownloading... {progress:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        logger.info(f"✅ Downloaded {downloaded} bytes")
        
        # Extract the ZIP file
        logger.info("Extracting documentation...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_path)
        
        # Clean up ZIP file
        zip_path.unlink()
        
        # Verify extraction
        html_files = list(target_path.rglob("*.html"))
        if not html_files:
            logger.error("No HTML files found after extraction")
            return False
        
        logger.info(f"✅ Extracted {len(html_files)} HTML files to {target_path}")
        return True
        
    except requests.RequestException as e:
        logger.error(f"❌ Failed to download documentation: {e}")
        return False
    except zipfile.BadZipFile as e:
        logger.error(f"❌ Failed to extract ZIP file: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    # Test the download
    import sys
    version = sys.argv[1] if len(sys.argv) > 1 else "stable"
    success = download_godot_html_docs(version=version)
    if not success:
        sys.exit(1)
