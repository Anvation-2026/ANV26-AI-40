import os
import sys
import time
import zipfile
import re
import requests

def download_tbx11k(dest_zip: str):
    os.makedirs(os.path.dirname(dest_zip), exist_ok=True)
    if os.path.exists(dest_zip) and os.path.getsize(dest_zip) >= 3300000000:
        print(f"Archive already exists and is full size: {dest_zip} ({os.path.getsize(dest_zip):,} bytes)")
        return

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    print("Requesting initial Google Drive download page...")
    url = "https://drive.google.com/uc?export=download&id=1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR"
    r1 = session.get(url)
    if r1.status_code != 200:
        raise RuntimeError(f"Failed to fetch initial download page: HTTP {r1.status_code}")

    uuid_match = re.search(r'name=["\']uuid["\']\s+value=["\']([^"\']+)["\']', r1.text)
    confirm_match = re.search(r'name=["\']confirm["\']\s+value=["\']([^"\']+)["\']', r1.text)

    uuid_val = uuid_match.group(1) if uuid_match else ""
    confirm_val = confirm_match.group(1) if confirm_match else "t"

    url2 = f"https://drive.usercontent.google.com/download?id=1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR&export=download&confirm={confirm_val}"
    if uuid_val:
        url2 += f"&uuid={uuid_val}"

    print(f"Connecting to stream download: {url2[:80]}...")
    r2 = session.get(url2, stream=True, timeout=60)
    if r2.status_code != 200:
        raise RuntimeError(f"Failed to initiate file stream: HTTP {r2.status_code}")

    total_bytes = int(r2.headers.get("content-length", 0))
    print(f"Total size: {total_bytes / (1024 * 1024):.2f} MB ({total_bytes:,} bytes)")

    downloaded = 0
    start_time = time.time()
    last_print = start_time
    chunk_size = 4 * 1024 * 1024  # 4 MB chunks

    temp_zip = dest_zip + ".part"
    with open(temp_zip, "wb") as f:
        for chunk in r2.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= 5.0 or downloaded == total_bytes:
                    elapsed = now - start_time
                    speed = downloaded / (elapsed + 1e-5) / (1024 * 1024)
                    pct = (downloaded / total_bytes * 100) if total_bytes > 0 else 0
                    print(f"Progress: {downloaded / (1024*1024):.1f}/{total_bytes / (1024*1024):.1f} MB ({pct:.1f}%) | Speed: {speed:.2f} MB/s | Elapsed: {elapsed:.1f}s")
                    last_print = now

    print(f"Download completed in {time.time() - start_time:.1f}s. Moving part to final path...")
    if os.path.exists(dest_zip):
        os.remove(dest_zip)
    os.rename(temp_zip, dest_zip)
    print(f"Saved: {dest_zip} ({os.path.getsize(dest_zip):,} bytes)")

def extract_tbx11k(zip_path: str, extract_dir: str):
    print(f"Extracting {zip_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    start_time = time.time()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        names = zip_ref.namelist()
        print(f"Zip archive contains {len(names)} entries. Extracting...")
        zip_ref.extractall(extract_dir)
    print(f"Extraction completed in {time.time() - start_time:.1f}s.")

if __name__ == "__main__":
    dest_zip = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "TBX11K.zip"))
    extract_target = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "TBX11K"))
    download_tbx11k(dest_zip)
    extract_tbx11k(dest_zip, extract_target)
    print("Done!")
