"""
STEP 1: Download the SMS Spam Collection dataset.

What is this dataset?
    - 5,574 SMS messages labeled as 'ham' (legitimate) or 'spam'
    - Collected by Tiago A. Almeida and José María Gómez Hidalgo
    - Source: UCI Machine Learning Repository
    - Format: Tab-separated, no header → label<TAB>message

Why this dataset?
    - Small (< 1MB) → fast to download and process
    - Already labeled → no manual annotation needed
    - Real-world SMS text → good for text classification
    - Well-studied benchmark → we can compare our results

What this script does:
    1. Downloads the ZIP from UCI's server
    2. Extracts the SMSSpamCollection file
    3. Renames it to .tsv for clarity
    4. Verifies the download with a quick row count
"""
import sys
import zipfile
import urllib.request
from pathlib import Path

# Add parent so we can import config
sys.path.insert(0, str(Path(__file__).parent))
import config


def download():
    """Download and extract the SMS Spam Collection dataset."""
    
    # Skip if already downloaded
    if config.RAW_FILE.exists():
        print(f"✓ Dataset already exists at: {config.RAW_FILE}")
        print(f"  Delete it and re-run if you want a fresh download.")
        return config.RAW_FILE

    print("=" * 50)
    print("STEP 1: Downloading SMS Spam Collection")
    print("=" * 50)
    print(f"  URL:  {config.DATASET_URL}")
    print(f"  Dest: {config.RAW_DIR}")
    
    # ── Download the ZIP ──
    zip_path = config.RAW_DIR / "smsspamcollection.zip"
    print("\n  Downloading...")
    urllib.request.urlretrieve(config.DATASET_URL, zip_path)
    print(f"  ✓ Downloaded ({zip_path.stat().st_size / 1024:.0f} KB)")
    
    # ── Extract ──
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # List contents so we know what's inside
        print(f"  ZIP contains: {zf.namelist()}")
        zf.extractall(config.RAW_DIR)
    
    # ── Rename to .tsv ──
    # The extracted file is named "SMSSpamCollection" (no extension)
    extracted = config.RAW_DIR / "SMSSpamCollection"
    if extracted.exists():
        extracted.rename(config.RAW_FILE)
        print(f"  ✓ Renamed to: {config.RAW_FILE.name}")
    elif config.RAW_FILE.exists():
        print(f"  ✓ File already at: {config.RAW_FILE.name}")
    else:
        # Try to find it with different names
        for f in config.RAW_DIR.iterdir():
            print(f"    Found: {f.name}")
        raise FileNotFoundError("Could not find extracted SMS collection file")
    
    # ── Clean up ZIP ──
    zip_path.unlink()
    # Also remove the readme if extracted
    readme = config.RAW_DIR / "readme"
    if readme.exists():
        readme.unlink()
    
    # ── Verify ──
    line_count = sum(1 for _ in open(config.RAW_FILE, encoding='latin-1'))
    print(f"\n  ✓ Verified: {line_count:,} lines in {config.RAW_FILE.name}")
    print(f"  (Expected ~5,574 lines)")
    
    return config.RAW_FILE


if __name__ == "__main__":
    download()
    print("\n✓ Step 1 complete!")
