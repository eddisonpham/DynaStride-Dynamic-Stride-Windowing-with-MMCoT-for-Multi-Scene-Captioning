import os
import json
import glob

# Find all folders in current directory that match "outputs*"
folders = sorted(glob.glob("outputs*"))

for outputDir in folders:
    json_files = sorted(glob.glob(os.path.join(outputDir, "*.json")))
    if not json_files:
        print(f"[{outputDir}] No .json files found.")
        continue

    for json_path in json_files:
        if not os.path.isfile(json_path):
            print(f"Missing file: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        
        for video_id, scenes in data.items():
            for scene_id, i in scenes.items():
                caption = i.get("predicted", "")
                if "<ANSWER>" not in caption:
                    continue

                parts = caption.split("<ANSWER>", 1)
                first_part = parts[0].strip()

                if first_part:
                    cleaned = first_part
                else:
                    after = parts[1]
                    cleaned = after.split("<", 1)[0].strip() if "<" in after else after.strip()

                if "." in cleaned:
                    cleaned = cleaned.split(".", 1)[0] + "."

                cleaned = cleaned.strip()
                i["predicted"] = cleaned

        # Write back updated file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Updated captions written back to {json_path}")
    else:
        print(f"Missing file: {json_path}")