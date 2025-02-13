import requests
import os

def download_tiktoken_files():
    """Download tiktoken encoding files directly from the source."""
    print("Starting tiktoken file download...")
    
    # Create output directory
    output_dir = "tiktoken_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define files to download
    files = {
        "cl100k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "p50k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
        "r50k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
    }
    
    for filename, url in files.items():
        output_path = os.path.join(output_dir, filename)
        print(f"\nDownloading {filename}...")
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            size = os.path.getsize(output_path)
            print(f"Successfully downloaded {filename} ({size} bytes)")
            
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
    
    # Verify downloaded files
    print("\nFiles in output directory:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        size = os.path.getsize(file_path)
        print(f"- {file} ({size} bytes)")

if __name__ == "__main__":
    download_tiktoken_files()