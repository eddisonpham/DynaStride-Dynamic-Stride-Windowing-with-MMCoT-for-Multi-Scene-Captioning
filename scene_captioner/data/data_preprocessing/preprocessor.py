import os, json, urllib.request, tarfile, cv2, sys
from collections import defaultdict
import string
import pickle

set_type = "training"
if len(sys.argv) == 2:
    set_type = sys.argv[1]
    if set_type != "training" and set_type != "validation" and set_type != "testing":
        print(f" Please enter a valid data set split, {set_type} is not valid")
        sys.exit(1)

# Using the argument sent in, we determine where we load in the data from
ann_path = "/workspace/scene_captioner/data/YouCookII/YouCookII/annotations/"
if set_type == "testing":
    ann_path += "youcookii_annotations_test_segments_only.json"
else:
    ann_path += "youcookii_annotations_trainval.json"
with open(ann_path, 'r') as f:
    data = json.load(f)

# locate the path for the raw videos
videos_dir = f"/workspace/scene_captioner/data/YouCookII/YouCookII/raw_videos/{set_type}/"

# build a list of recipe types + the name of the video for video path
type_paths = []
if set_type == "testing":
    with open('/workspace/scene_captioner/data/YouCookII/YouCookII/splits/test_list.txt', 'r') as f:
        for line in f:
            line = line.strip()
            type_paths.append(line)
elif set_type == "training":
    with open('/workspace/scene_captioner/data/YouCookII/YouCookII/splits/train_list.txt', 'r') as f:
        for line in f:
            line = line.strip()
            type_paths.append(line)
elif set_type == "validation":
   with open('/workspace/scene_captioner/data/YouCookII/YouCookII/splits/val_list.txt', 'r') as f:
        for line in f:
            line = line.strip()
            type_paths.append(line) 

# normalize the annotations (METEOR and CIDEr can do this on their own, but this is best to avoid inconsistencies)
def normalize_caption(caption):
    caption = caption.lower() # make it lowercase
    caption = caption.translate(str.maketrans('', '', string.punctuation)) # removes punctuation
    caption = ' '.join(caption.split()) # strip the spaces
    return caption

# build a ground-truth dictionary from the annotations
references = defaultdict(list)
count = 0
for vid, vid_data in data['database'].items():
    if vid_data['subset'] != set_type:
        continue

    # getting the video path
    end_path = ""
    for path in type_paths:
        if vid in path:
            end_path = path
    vid_path = os.path.join(videos_dir, f"{end_path}.mp4")
    
    # obtaining the frames per second for the video
    fps = 30
    if os.path.exists(vid_path):
        print(vid_path)
        count += 1
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    if fps <= 0:
        fps = 30
    
    # filling out the dictionary
    for index, annotation in enumerate(vid_data['annotations']):
        seg_id = f"{vid}_{index}"
        caption = normalize_caption(annotation['sentence'])
        
        startTime , endTime = annotation['segment']
        startFrame = int(float(startTime) * fps)
        endFrame = int(float(endTime) * fps)
        
        references[seg_id].append({
          'caption': caption,
          'startTime': float(startTime),
          'endTime': float(endTime),
          'startFrame': startFrame,
          'endFrame': endFrame,
        })

# we can make dummy predictions for now, which will later be replaced by our model's output
predictions = { seg_id: refs[0] for seg_id, refs in references.items() }

# saving the dictionary
if set_type == "testing":
    with open("/workspace/scene_captioner/data/YouCookII/YouCookII/saved_references/youcook2_test_refs.pkl", "wb") as f:
        pickle.dump(dict(references), f)
    with open("/workspace/scene_captioner/data/YouCookII/YouCookII/saved_references/youcook2_test_preds.pkl", "wb") as f:
        pickle.dump(predictions, f)
    print(f"Total validation segments: {len(references)}")
elif set_type == "training":
    with open("/workspace/scene_captioner/data/YouCookII/YouCookII/saved_references/youcook2_train_refs.pkl", "wb") as f:
        pickle.dump(dict(references), f)
    with open("/workspace/scene_captioner/data/YouCookII/YouCookII/saved_references/youcook2_train_preds.pkl", "wb") as f:
        pickle.dump(predictions, f)
    print(f"Total validation segments: {len(references)}")
elif set_type == "validation":
    with open("/workspace/scene_captioner/data/YouCookII/YouCookII/saved_references/youcook2_val_refs.pkl", "wb") as f:
        pickle.dump(dict(references), f)
    with open("/workspace/scene_captioner/data/YouCookII/YouCookII/saved_references/youcook2_val_preds.pkl", "wb") as f:
        pickle.dump(predictions, f)
    print(f"Total validation segments: {len(references)}")