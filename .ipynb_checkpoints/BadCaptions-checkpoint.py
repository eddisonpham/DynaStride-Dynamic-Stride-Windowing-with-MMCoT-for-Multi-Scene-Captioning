import os
import json
import glob

total_count = 0

# Find all folders matching "outputs*"
folders = glob.glob("outputs*")

for output_dir in folders:
    json_files = sorted(glob.glob(os.path.join(output_dir, "*.json")))
    if not json_files:
        print(f"[{output_dir}] No .json files found.")
        continue

    folder_count = 0

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Iterate through each video and scene (scenes is a list of dicts)
        for video_id, scenes in data.items():
            if not isinstance(scenes, list):
                continue
            for scene in scenes:
                if "<ANSWER>" in scene.get("predicted", ""):
                    total_count += 1
                    folder_count += 1

    print(f"{output_dir}: {folder_count} defective entries")

print(f"\nTOTAL defective entries: {total_count}")
