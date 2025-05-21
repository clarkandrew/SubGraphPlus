import os
import sys
import argparse
import logging
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path so script can import app modules
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("download_models")

# Constants
DEFAULT_MODEL_DIR = os.path.join(parent_dir, "models")
MLP_MODEL_URL = "https://huggingface.co/SubgraphRAG/mlp-scorer/resolve/main/mlp_model.zip"
LOCAL_MODELS_ZIP = os.path.join(parent_dir, "models.zip")


def download_file(url, local_filename=None, chunk_size=8192):
    """
    Download a file from a URL with progress bar
    
    Args:
        url: URL to download
        local_filename: Local filename to save to
        chunk_size: Size of chunks to download
        
    Returns:
        Path to downloaded file
    """
    if local_filename is None:
        local_filename = url.split('/')[-1]
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(local_filename, 'wb') as f, tqdm(
                desc=f"Downloading {os.path.basename(local_filename)}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        bar.update(len(chunk))
    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        if os.path.exists(local_filename):
            os.remove(local_filename)
        return None
    
    return local_filename


def extract_zip(zip_file, extract_to):
    """
    Extract a zip file
    
    Args:
        zip_file: Path to zip file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(extract_to, exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get number of files for progress bar
            file_count = len(zip_ref.namelist())
            logger.info(f"Extracting {file_count} files to {extract_to}")
            
            # Extract with progress bar
            for file in tqdm(zip_ref.namelist(), desc="Extracting files"):
                zip_ref.extract(file, extract_to)
        
        return True
    except zipfile.BadZipFile:
        logger.error(f"Invalid zip file: {zip_file}")
        return False
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return False


def check_local_models_zip():
    """Check if models.zip exists locally"""
    if os.path.exists(LOCAL_MODELS_ZIP):
        logger.info(f"Found local models.zip: {LOCAL_MODELS_ZIP}")
        return LOCAL_MODELS_ZIP
    return None


def download_models(model_dir=DEFAULT_MODEL_DIR, force=False):
    """
    Download and extract MLP model
    
    Args:
        model_dir: Directory to save models to
        force: Force download even if models already exist
        
    Returns:
        True if successful, False otherwise
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if models already exist
    mlp_dir = os.path.join(model_dir, "mlp")
    if os.path.exists(mlp_dir) and not force:
        logger.info(f"MLP model already exists at {mlp_dir}")
        return True
    
    # Check if we have a local models.zip first
    local_zip = check_local_models_zip()
    
    if not local_zip:
        # Download from HuggingFace
        logger.info(f"Downloading MLP model from {MLP_MODEL_URL}")
        download_path = download_file(MLP_MODEL_URL, os.path.join(model_dir, "mlp_model.zip"))
        if not download_path:
            logger.error("Failed to download MLP model")
            return False
    else:
        # Use local zip file
        download_path = local_zip
    
    # Extract model
    logger.info(f"Extracting MLP model to {model_dir}")
    if not extract_zip(download_path, model_dir):
        logger.error("Failed to extract MLP model")
        return False
    
    # Clean up zip file if it was downloaded (not if it was the local one)
    if download_path != LOCAL_MODELS_ZIP and os.path.exists(download_path):
        os.remove(download_path)
    
    logger.info("MLP model downloaded and extracted successfully")
    return True


def main():
    """Main entry point for download_models script"""
    parser = argparse.ArgumentParser(description="SubgraphRAG+ Model Downloader")
    parser.add_argument("--dir", default=DEFAULT_MODEL_DIR, help="Model directory")
    parser.add_argument("--force", action="store_true", help="Force download even if models exist")
    
    args = parser.parse_args()
    
    if download_models(args.dir, args.force):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())