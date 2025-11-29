"""YouCook2 data preprocessor."""
import os
import json
import urllib.request
import tarfile
import cv2
import sys
from collections import defaultdict
import string
import pickle

WORKSPACE = os.path.abspath(os.path.join(os.getcwd(), ".."))
SCENE_CAPTIONER_DATA = os.path.join(WORKSPACE, "YouCookII")

def normalize_caption(caption):
    """Normalize caption for evaluation."""
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = ' '.join(caption.split())
    return caption


def preprocess_youcook2():
    """Preprocess YouCook2 dataset annotations."""
    ann_path = os.path.join(SCENE_CAPTIONER_DATA, "youcookii_annotations_trainval.json")
    
    with open(ann_path, 'r') as f:
        data = json.load(f)

    videos_dir = os.path.join(SCENE_CAPTIONER_DATA, f"raw_videos/validation/")

    type_paths = []
    splits_dir = os.path.join(SCENE_CAPTIONER_DATA, "splits")
    split_file = os.path.join(splits_dir, "val_list.txt")
    
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            type_paths.append(line)

    references = defaultdict(list)
    count = 0
    for vid, vid_data in data['database'].items():
        if vid_data['subset'] != "validation":
            continue

        end_path = ""
        for path in type_paths:
            if vid in path:
                end_path = path
        vid_path = os.path.join(videos_dir, f"{end_path}.mp4")
        
        fps = 30
        if os.path.exists(vid_path):
            print(vid_path)
            count += 1
            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        if fps <= 0:
            fps = 30
        
        for index, annotation in enumerate(vid_data['annotations']):
            seg_id = f"{vid}_{index}"
            caption = normalize_caption(annotation['sentence'])
            
            startTime, endTime = annotation['segment']
            startFrame = int(float(startTime) * fps)
            endFrame = int(float(endTime) * fps)
            
            references[seg_id].append({
                'caption': caption,
                'startTime': float(startTime),
                'endTime': float(endTime),
                'startFrame': startFrame,
                'endFrame': endFrame,
            })

    predictions = {seg_id: refs[0] for seg_id, refs in references.items()}

    saved_refs_dir = os.path.join(SCENE_CAPTIONER_DATA, "saved_references")
    os.makedirs(saved_refs_dir, exist_ok=True)
    
    refs_path = os.path.join(saved_refs_dir, f"youcook2_validation_refs.pkl")
    preds_path = os.path.join(saved_refs_dir, f"youcook2_validation_preds.pkl")
    
    with open(refs_path, "wb") as f:
        pickle.dump(dict(references), f)
    
    with open(preds_path, "wb") as f:
        pickle.dump(predictions, f)
    
    print(f"Total validation segments: {len(references)}")


if __name__ == "__main__":
    preprocess_youcook2()

