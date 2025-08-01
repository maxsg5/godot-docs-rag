import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Custom progress bar for download progress"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_godot_docs():
    """
    Download and extract Godot documentation from the nightly build.
    """
    # Create data/raw directory if it doesn't exist
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URL for the Godot docs zip file
    docs_url = "https://nightly.link/godotengine/godot-docs/workflows/build_offline_docs/master/godot-docs-html-stable.zip"
    zip_filename = data_dir / "godot-docs-html-stable.zip"
    
    print(f"Downloading Godot documentation from: {docs_url}")
    print(f"Saving to: {zip_filename}")
    
    try:
        # Download the zip file with progress bar
        print("📥 Starting download...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(docs_url, zip_filename, reporthook=t.update_to)
        print(f"✅ Download completed: {zip_filename}")
        
        # Extract the zip file with progress bar
        print(f"📦 Extracting {zip_filename}...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            # Get list of files to extract
            file_list = zip_ref.namelist()
            
            # Extract with progress bar
            with tqdm(total=len(file_list), desc="Extracting", unit="files") as pbar:
                for file in file_list:
                    zip_ref.extract(file, data_dir)
                    pbar.update(1)
        
        print(f"✅ Extraction completed to: {data_dir}")
        
        # Optional: Remove the zip file after extraction
        print("🧹 Cleaning up...")
        os.remove(zip_filename)
        print("✅ Zip file removed after extraction")
        
        # List the contents of the data/raw directory
        print("\nContents of data/raw directory:")
        for item in data_dir.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}/")
            else:
                print(f"  📄 {item.name}")
                
    except Exception as e:
        print(f"Error downloading or extracting documentation: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = download_and_extract_godot_docs()
    if success:
        print("\n✅ Godot documentation download and extraction completed successfully!")
    else:
        print("\n❌ Failed to download and extract Godot documentation.")
