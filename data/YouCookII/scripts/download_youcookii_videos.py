import os
import yt_dlp
import sys

DATASET_ROOT = '../raw_videos'
VID_FILE_PATH = '../splits/val_list.txt'
SPLIT_NAME = 'validation'

os.makedirs(DATASET_ROOT, exist_ok=True)
os.makedirs(os.path.join(DATASET_ROOT, SPLIT_NAME), exist_ok=True)

missing_vid_lst = []

def download_video_no_audio_mp4(url, out_prefix):
    """
    Download only the video stream as mp4 using yt_dlp. No audio is downloaded or merged.
    """
    ydl_opts = {
        "format": "bestvideo[ext=mp4]/bestvideo+bestaudio/best",
        "outtmpl": out_prefix + ".%(ext)s",
        "quiet": True,
        "noprogress": True,
        "merge_output_format": "mp4",
        "postprocessors": [{
            "key": "FFmpegVideoRemuxer",
            "preferedformat": "mp4"
        }],
        "postprocessor_args": [
            "-an"
        ],
        "extractor_args": {
            "youtube": ["player_client=default"]
        }
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"[WARNING] yt-dlp error for url {url}: {e}", file=sys.stderr)
        return False

with open(VID_FILE_PATH) as f:
    lines = f.readlines()

for line in lines:
    recipe_type, vid_name = line.strip().split('/')

    recipe_dir = os.path.join(DATASET_ROOT, SPLIT_NAME, recipe_type)
    os.makedirs(recipe_dir, exist_ok=True)

    vid_url = f"https://www.youtube.com/watch?v={vid_name}"
    vid_prefix = os.path.join(recipe_dir, vid_name)

    target_mp4 = vid_prefix + '.mp4'
    if os.path.exists(target_mp4):
        print(f"[INFO] Found {SPLIT_NAME} video {vid_name} as mp4, skipping download.")
        continue

    for ext in ['.mkv', '.webm']:
        alt = vid_prefix + ext
        if os.path.exists(alt):
            os.remove(alt)

    success = download_video_no_audio_mp4(vid_url, vid_prefix)

    downloaded = os.path.exists(target_mp4)

    if success and downloaded:
        print(f"[INFO] Downloaded {SPLIT_NAME} video {vid_name} (NO AUDIO) as mp4")
    else:
        missing_vid_lst.append(f"{SPLIT_NAME}/{line}")
        print(f"[INFO] Cannot download {SPLIT_NAME} video {vid_name} as mp4")

with open('missing_videos.txt', 'w') as missing_vid:
    for line in missing_vid_lst:
        missing_vid.write(line)

os.system("find ../raw_videos -name '*.f*' -delete")
