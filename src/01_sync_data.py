"""
Downloads all .wav files from Azure Blob Storage to a local `raw_audio/` directory.

Features:
 - Multi-threaded for extremely fast syncing (20 connections).
 - Skips files that are already downloaded (resumable).
"""

import os
import sys
import time
# Ensure src/ siblings are importable regardless of working directory
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from tqdm.auto import tqdm

from config import RAW_DIR

# Load environment variables
load_dotenv()

CONTAINER_NAME = "wav-files"
PREFIXES = ["RAVDESS/", "CREMAD/", "TESS/", "SAVEE/", "ASVP-ESD/"]


def download_blob(blob_service, blob_name: str, local_path: Path) -> tuple[bool, str]:
    """Download a single blob to the local file system if it doesn't exist."""
    try:
        if local_path.exists():
            return False, ""  # skipped
        
        # Ensure parent directories exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        blob_client = blob_service.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        with open(local_path, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
            
        return True, ""  # downloaded
    except Exception as e:
        return False, f"Failed {blob_name}: {e}"


def main():
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set in .env")

    print("Connecting to Azure...")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(CONTAINER_NAME)

    print("Listing remote blobs...")
    download_tasks = []
    
    # 1. Gather all files we need to download
    for prefix in PREFIXES:
        blobs = container_client.list_blobs(name_starts_with=prefix)
        for blob in blobs:
            if blob.name.endswith(".wav"):
                # Clean windows paths if any snuck in
                clean_name = blob.name.replace("\\", "/")
                local_path = RAW_DIR / clean_name
                download_tasks.append((clean_name, local_path))
                
    total_files = len(download_tasks)
    if total_files == 0:
        print("No .wav files found on Azure matching prefixes!")
        return
        
    print(f"Found {total_files} files. Checking local cache...")

    downloaded_count = 0
    skipped_count = 0
    errors = []

    t0 = time.time()
    
    # 2. ThreadPool download
    
    with ThreadPoolExecutor(max_workers=20) as executor: # download 20 files at same time
        futures = {
            executor.submit(download_blob, blob_service, blob_name, local_path): blob_name
            for blob_name, local_path in download_tasks
        }
        
        for future in tqdm(as_completed(futures), total=total_files, desc="Syncing Data"):
            was_downloaded, err = future.result()
            if err:
                errors.append(err)
            elif was_downloaded:
                downloaded_count += 1
            else:
                skipped_count += 1

    elapsed = (time.time() - t0) / 60
    
    # 3. Summary
    print("\n" + "="*40)
    print(" Sync Complete")
    print("="*40)
    print(f"Time Taken:  {elapsed:.1f} minutes")
    print(f"Total Files: {total_files}")
    print(f"Downloaded:  {downloaded_count}")
    print(f"Skipped:     {skipped_count} (already on disk)")
    if errors:
        print(f"Errors:      {len(errors)}")
        for e in errors[:5]:
            print(f"  - {e}")
        if len(errors) > 5:
            print("  - ...")

if __name__ == "__main__":
    main()
