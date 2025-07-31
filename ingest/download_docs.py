"""
Download Godot documentation from GitHub
"""

import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_godot_docs(
    target_dir="data/raw/godot-docs",
    branch="4.4",
    repo_url="https://github.com/godotengine/godot-docs.git"
) -> bool:
    """
    Download Godot documentation from GitHub
    
    Args:
        target_dir: Directory to clone the docs into
        branch: Git branch to checkout
        repo_url: Repository URL
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        target_path = Path(target_dir)
        
        # Create parent directory
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        if target_path.exists():
            logger.info("Godot docs already exist. Pulling latest changes...")
            # Pull latest changes
            result = subprocess.run(
                ["git", "pull", "origin", branch],
                cwd=target_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.warning(f"Git pull failed: {result.stderr}")
                # Try to re-clone
                logger.info("Attempting to re-clone repository...")
                subprocess.run(["rm", "-rf", str(target_path)], check=True)
                return download_godot_docs(target_dir, branch, repo_url)
        else:
            logger.info("Cloning Godot docs repository...")
            result = subprocess.run(
                ["git", "clone", repo_url, str(target_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Git clone failed: {result.stderr}")
                return False
        
        # Checkout specific branch
        logger.info(f"Checking out branch: {branch}")
        result = subprocess.run(
            ["git", "checkout", branch],
            cwd=target_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Git checkout failed: {result.stderr}")
            return False
        
        # Verify docs were downloaded
        conf_py = target_path / "conf.py"
        if not conf_py.exists():
            logger.error(f"conf.py not found in {target_path}")
            return False
        
        logger.info(f"✅ Godot documentation downloaded successfully to {target_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download Godot docs: {e}")
        return False


if __name__ == "__main__":
    download_godot_docs()
