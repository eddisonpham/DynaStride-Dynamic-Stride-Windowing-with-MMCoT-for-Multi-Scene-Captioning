import os
import json
import numpy as np
BASE_DIR = "/workspace/scene_captioner/LLaMACoT/results/mistral"
FOLDERS = ["5"]

def process_folder(folder_path, output_file):
    metrics_data = {}

    # Read all JSON files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file), "r") as f:
                data = json.load(f)

                # Loop through each metric in the JSON file
                for key, value in data.items():
                    if isinstance(value, (int, float)):  # only numeric metrics
                        metrics_data.setdefault(key, []).append(value)

    # Compute averages and stds
    with open(output_file, "w") as out:
        for key, values in metrics_data.items():
            avg = np.mean(values)
            std = np.std(values)
            out.write(f"{key}: avg = {avg:.4f}, std = {std:.4f}\n")

def main():
    for folder in FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        output_file = os.path.join(BASE_DIR, f"{folder}_results.txt")
        process_folder(folder_path, output_file)
        print(f"Results saved for {folder} -> {output_file}")

if __name__ == "__main__":
    main()
