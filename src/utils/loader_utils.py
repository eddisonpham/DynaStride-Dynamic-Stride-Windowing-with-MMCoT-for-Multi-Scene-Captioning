"""Data loading utilities for video scene captioning experiments."""
import os
import json
import glob
import re
import pathlib
from typing import List, Dict


def read_sample_file(path: str) -> List[str]:
    """Read sample file, filtering out empty lines and comments."""
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def get_ground_truth(gt_path: str) -> Dict[str, Dict[int, str]]:
    """Load ground truth annotations from YouCookII JSON format."""
    with open(gt_path, "r", encoding="utf-8") as f:
        db = json.load(f).get("database", {})
    gt: Dict[str, Dict[int, str]] = {}
    for vid, meta in db.items():
        inner: Dict[int, str] = {}
        for a in meta.get("annotations", []):
            sid = int(a.get("id"))
            inner[sid] = a.get("sentence", "").strip()
        gt[vid] = inner
    return gt


def find_scene_folders(video_root: str) -> List[str]:
    """Find scene folders containing images in a video directory."""
    if not os.path.isdir(video_root):
        return []
    subs = [p for p in glob.glob(os.path.join(video_root, "*")) if os.path.isdir(p)]
    keep = []
    for s in subs:
        if glob.glob(os.path.join(s, "*.jpg")) or glob.glob(os.path.join(s, "*.jpeg")):
            keep.append(s)
    
    def natural_sort_key(p):
        name = os.path.basename(p)
        m = re.fullmatch(r"\d+", name)
        return (0, int(name)) if m else (1, name)
    
    return sorted(keep, key=natural_sort_key)


def get_data_paths() -> Dict[str, str]:
    """Get data paths."""
    workspace = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    youcookii_path = os.path.join(workspace, "data", "YouCookII")
    
    base_dir = os.path.join(youcookii_path, "raw_videos/validation")
    sample_file = os.path.join(workspace, "constants/sampled_videos.txt")
    gt_file = os.path.join(youcookii_path, "youcookii_annotations_trainval.json")
    output_dir = os.path.join(workspace, "outputs")

    os.makedirs(output_dir, exist_ok=True)
    
    return {
        "baseDir": base_dir,
        "sampleFile": sample_file,
        "gtFile": gt_file,
        "outputDir": output_dir,
    }

