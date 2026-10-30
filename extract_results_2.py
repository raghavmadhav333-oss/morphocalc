import os
import sys
import glob
import zipfile

candidate_paths = [
    r"d:\Research\morphocalc\results (2).zip",
    r"d:\Research\morphocalc\results(2).zip",
    r"d:\Research\results (2).zip",
    r"d:\Research\results(2).zip",
    os.path.expanduser(r"~\Downloads\results (2).zip"),
    os.path.expanduser(r"~\Downloads\results(2).zip"),
    os.path.expanduser(r"~\Downloads\results.zip"),
    os.path.expanduser(r"~\Downloads\results (1).zip"),
]

# Also check Downloads for any results*.zip
downloads_pattern = os.path.expanduser(r"~\Downloads\results*.zip")
for p in glob.glob(downloads_pattern):
    if p not in candidate_paths:
        candidate_paths.append(p)

print("[*] Searching for results (2) zip...")
target_zip = None
for path in candidate_paths:
    if os.path.exists(path):
        print(f"  Found: {path} ({os.path.getsize(path)/(1024*1024):.2f} MB)")
        if "results (2)" in path or "results(2)" in path:
            target_zip = path
            break
        elif target_zip is None:
            target_zip = path

if not target_zip:
    print("[-] Could not find any results zip file in expected locations.")
    sys.exit(1)

print(f"\n[*] Target zip selected: {target_zip}")

# Extract destination
extract_dir = r"d:\Research\morphocalc\kaggle_results_2"
os.makedirs(extract_dir, exist_ok=True)

print(f"[*] Extracting to: {extract_dir}")
with zipfile.ZipFile(target_zip, 'r') as zf:
    print("\n--- FILES IN ARCHIVE ---")
    for info in zf.infolist():
        print(f"  - {info.filename} ({info.file_size / (1024*1024):.2f} MB)")
    zf.extractall(extract_dir)

print("\n[✓] Successfully extracted all files!")
