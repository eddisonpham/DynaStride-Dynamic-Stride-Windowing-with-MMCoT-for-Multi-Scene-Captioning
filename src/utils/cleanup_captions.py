"""Cleanup script for caption JSON files."""
import os
import json
import glob


def cleanup_captions_in_file(json_path: str) -> bool:
    """Clean up captions in a single JSON file.
    
    Args:
        json_path: Path to JSON file to clean
        
    Returns:
        True if file was updated, False otherwise
    """
    if not os.path.isfile(json_path):
        print(f"Missing file: {json_path}")
        return False

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = False
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
            if cleaned != caption:
                i["predicted"] = cleaned
                updated = True

    if updated:
        # Write back updated file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Updated captions written back to {json_path}")
        return True
    
    return False


def main():
    """Main entry point for cleanup script."""
    # Find all output directories
    output_dirs = glob.glob("outputs*")
    if not output_dirs:
        print("No outputs* directories found.")
        return

    for output_dir in output_dirs:
        json_files = sorted(glob.glob(os.path.join(output_dir, "*.json")))
        if not json_files:
            print(f"[{output_dir}] No .json files found.")
            continue

        for json_path in json_files:
            cleanup_captions_in_file(json_path)


if __name__ == "__main__":
    main()

