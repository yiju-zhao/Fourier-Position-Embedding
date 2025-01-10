import requests
import argparse
from tqdm import tqdm
import os

def download_file(url, local_filepath):
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(local_filepath, 'wb') as f, tqdm(
            desc=local_filepath,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
    return local_filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download file from URL")
    parser.add_argument(
        "--url", 
        default=None,
        required=True,
        help="URL to download file"
    )
    parser.add_argument(
        "--local_filepath",
        default=None,
        help="Local filename to save")
    args = parser.parse_args()
    
    if args.local_filepath is None:
        args.local_filepath = os.path.join("/root/data/OLMo", args.url.split("https://olmo-data.org/")[-1])
        local_dir = os.path.dirname(args.local_filepath)
        
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
    
    download_file(args.url, args.local_filepath)
    print(f"Data saved in: {args.local_filepath}")
