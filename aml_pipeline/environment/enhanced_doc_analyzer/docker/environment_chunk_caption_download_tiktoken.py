#!/usr/bin/env python
# Script to download tiktoken encoding files

import requests
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tiktoken_downloader")

def download_tiktoken_files():
    """Download tiktoken encoding files directly from OpenAI source."""
    logger.info("Starting tiktoken file download...")
    
    # Create output directory
    output_dir = "./tiktoken_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define files to download
    files = {
        "cl100k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "p50k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
        "r50k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
    }
    
    for filename, url in files.items():
        output_path = os.path.join(output_dir, filename)
        logger.info(f"Downloading {filename} from {url}...")
        
        try:
            # Download with timeout and retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()  # Raise exception for bad status codes
                    break
                except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying...")
                    else:
                        raise
            
            # Save file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            size = os.path.getsize(output_path)
            logger.info(f"Successfully downloaded {filename} ({size} bytes)")
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {str(e)}")
    
    # Verify downloaded files
    logger.info("Files in output directory:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        size = os.path.getsize(file_path)
        logger.info(f"- {file} ({size} bytes)")

if __name__ == "__main__":
    download_tiktoken_files()