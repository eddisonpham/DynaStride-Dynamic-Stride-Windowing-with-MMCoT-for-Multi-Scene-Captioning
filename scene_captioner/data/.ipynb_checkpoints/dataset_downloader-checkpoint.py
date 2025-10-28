import os
import argparse
import concurrent.futures
import subprocess
import random
import json
import shutil
import tarfile
import urllib.request

def download_coin(output_path="./videos"):
    repo_url = "https://github.com/coin-dataset/annotations.git"
    repo_dir = "coin_annotations"
    
    # Clone the repo if it doesn't exist
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    
    # Ensure youtube-dl is installed
    try:
        # using yt-dlp instead for a more updated version
        subprocess.run(["yt-dlp", "--version"], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # again using yt-dlp
        subprocess.run(["pip", "install", "yt-dlp"], check=True)
    
    # Load COIN.json
    json_path = os.path.join(repo_dir, "COIN.json")
    with open(json_path, "r") as f:
        data = json.load(f)["database"]
    
    # Sample 50% of videos
    all_keys = list(data.keys())
    #sample_size = len(all_keys) /for / 2
    # changing to make the sample size 5 for my purposes, we can change this back later, just experimenting
    sample_size = 5
    sampled_keys = set(random.sample(all_keys, sample_size))

    # Create output path
    #output_path = './videos' # changed this from earlier
    os.makedirs(output_path, exist_ok=True)

    for youtube_id in sampled_keys:
        info = data[youtube_id]
        type_name = info['recipe_type']
        url = info['video_url']
        vid_dir = os.path.join(output_path, str(type_name))
        os.makedirs(vid_dir, exist_ok=True)

        # Build youtube-dl command (best but not better than 480p)
        output_template = os.path.join(vid_dir, f"{youtube_id}.mp4")
        cmd = f"yt-dlp -f 'best[height<=480]' -o \"{output_template}\" {url}"
        #os.system(cmd)
        # added the following struct, may be safer
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {youtube_id}: {url}")

    print(f"Downloaded {sample_size} videos to {output_path}")

def download_youcook():
    # Target directory for extraction
    target_dir = os.path.join("YouCookII")
    os.makedirs(target_dir, exist_ok=True)

    # URL and local file path
    url = "http://youcook2.eecs.umich.edu/static/YouCookII/YouCookII.tar.gz"
    archive_path = os.path.join(target_dir, "YouCookII.tar.gz")

    # Download the tar.gz archive if not already downloaded
    if not os.path.exists(archive_path):
        print("Downloading YouCookII.tar.gz...")
        urllib.request.urlretrieve(url, archive_path)
        print("Download completed.")
    else:
        print("YouCookII.tar.gz already exists, skipping download.")

    # Extract the archive
    print("Extracting archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
    print("Extraction completed.")

    # Remove archive to save space
    os.remove(archive_path)
    print("Removed archive to save disk space.")

def download_lvd2m():
    print("Downloading LVD-2M (HDVG-300k)...")

    output_path = "./videos/LVD-2M-HDVG300k"
    os.makedirs(output_path, exist_ok=True)

    repo_url = "https://github.com/SilentView/LVD-2M.git"
    repo_dir = "LVD-2M"

    # Clone repo if missing
    if not os.path.exists(repo_dir):
        try:
            subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        except subprocess.CalledProcessError as e:
            print("Failed to clone LVD-2M repo:", e)
            return

    # find a CSV that looks like the 300k split (be permissive)
    import glob, csv
    csv_candidates = glob.glob(os.path.join(repo_dir, "**", "*.csv"), recursive=True)
    chosen_csv = None
    for c in csv_candidates:
        name = os.path.basename(c).lower()
        if "300" in name and ("hdvg" in name or "300k" in name or "720p" in name):
            chosen_csv = c
            break
    if chosen_csv is None and csv_candidates:
        # fallback to first CSV in the repo
        chosen_csv = csv_candidates[0]

    if chosen_csv is None:
        print("No CSV metadata found in the LVD-2M repo. Aborting.")
        return

    print(f"Using metadata CSV: {chosen_csv}")

    # ensure yt-dlp is available
    try:
        subprocess.run(["yt-dlp", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        try:
            subprocess.run(["pip", "install", "yt-dlp"], check=True)
        except Exception as e:
            print("Failed to install yt-dlp:", e)
            return

    # read CSV and download each entry
    with open(chosen_csv, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # try a bunch of common field names for id/url
            youtube_id = None
            for k in ("youtube_id", "video_id", "yt_id", "ytid", "id"):
                if k in row and row[k].strip():
                    youtube_id = row[k].strip()
                    break

            url = None
            for k in ("video_url", "url", "youtube_url", "youtube"):
                if k in row and row[k].strip():
                    url = row[k].strip()
                    break

            # if no url but have id, build a YouTube watch URL
            if not url and youtube_id:
                url = f"https://www.youtube.com/watch?v={youtube_id}"

            if not youtube_id and url:
                # try to extract id from url (simple heuristic)
                # supports full watch?v= or youtu.be links
                if "v=" in url:
                    youtube_id = url.split("v=")[-1].split("&")[0]
                elif "youtu.be/" in url:
                    youtube_id = url.split("youtu.be/")[-1].split("?")[0]

            if not youtube_id or not url:
                print("Skipping malformed row (no id/url):", {k: row.get(k) for k in row})
                continue

            youtube_id = youtube_id.strip()
            # create an output template so yt-dlp picks correct extension
            output_template = os.path.join(output_path, f"{youtube_id}.%(ext)s")
            # skip if we already have a matching file (any extension)
            existing = glob.glob(os.path.join(output_path, youtube_id + ".*"))
            if existing:
                print(f"Already exists, skipping: {youtube_id}")
                continue

            cmd = ["yt-dlp", "-f", "best[height<=480]", "-o", output_template, url]
            try:
                subprocess.run(cmd, check=True)
                print(f"Downloaded {youtube_id}")
            except subprocess.CalledProcessError:
                print(f"Failed to download {youtube_id}: {url}")

    print(f"Finished downloading LVD-2M (HDVG-300k) to {output_path}")

def download_movienet():
    print("Downloading MovieNet...")
    # actual download code
    pass

def download_ssv2():
    print("Downloading Something-SomethingV2...")
    # actual download code
    pass

def download_charades():
    print("Downloading Charades...")
    # actual download code
    pass

def download_vdc():
    print("Downloading VDC...")
    # actual download code
    pass

dataset_mapping = {
    "COIN": download_coin,
    "YouCookII": download_youcook,
    "LVD-2M": download_lvd2m,
    "MovieNet": download_movienet,
    "Something-SomethingV2": download_ssv2,
    "Charades": download_charades,
    "VDC": download_vdc
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", nargs="*", help="List of datasets to download", default=[])
    args = parser.parse_args()

    # Validate dataset names
    for name in args.download:
        if name not in dataset_mapping:
            raise ValueError(f"Unknown dataset name: {name}")

    # Download in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(dataset_mapping[name]) for name in args.download]
        concurrent.futures.wait(futures)
    
if __name__ == "__main__":
    main()