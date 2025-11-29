import os
import json
import numpy as np

def process_folder(folder_path, output_file):
    """
    Process all JSON files in a folder and compute averages and standard deviations.
    """
    metrics_data = {}

    # Read all JSON files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            json_path = os.path.join(folder_path, file)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Loop through each metric in the JSON file
                for key, value in data.items():
                    if isinstance(value, (int, float)):  # only numeric metrics
                        # Handle NaN values
                        if not np.isnan(value):
                            metrics_data.setdefault(key, []).append(value)
            except Exception as e:
                print(f"Warning: Could not read {json_path}: {e}")
                continue

    # Compute averages and stds
    with open(output_file, "w", encoding="utf-8") as out:
        for key in sorted(metrics_data.keys()):
            values = metrics_data[key]
            if values:
                avg = np.mean(values)
                std = np.std(values)
                out.write(f"{key}: avg = {avg:.4f}, std = {std:.4f}\n")

def process_evaluation_results(eval_results_dir):
    """
    Process evaluation results for a single results type (baseline or dynastride).
    """
    print(f"\n{'='*60}")
    print(f"Processing: {eval_results_dir}")
    print(f"{'='*60}")
    
    if not os.path.isdir(eval_results_dir):
        print(f"  Warning: Directory not found: {eval_results_dir}")
        return
    
    # Loop through all model directories
    for model_name in sorted(os.listdir(eval_results_dir)):
        model_dir = os.path.join(eval_results_dir, model_name)
        
        # Skip if not a directory or if it's a hidden file or the script itself
        if not os.path.isdir(model_dir) or model_name.startswith('.'):
            continue
        
        print(f"\nProcessing model: {model_name}")
        
        # Find all frame_sparsity subdirectories
        frame_sparsities = []
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if it looks like a frame_sparsity folder (contains numbers)
                if item.isdigit() or (item.replace('_', '').replace('-', '').isdigit()):
                    frame_sparsities.append(item)
        
        if not frame_sparsities:
            print(f"  No frame_sparsity folders found for {model_name}")
            continue
        
        # Process each frame_sparsity folder
        for frame_sparsity in sorted(frame_sparsities, key=lambda x: int(x) if x.isdigit() else 0):
            folder_path = os.path.join(model_dir, frame_sparsity)
            output_file = os.path.join(model_dir, f"{frame_sparsity}_results.txt")
            
            # Check if folder has JSON files
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            if not json_files:
                print(f"  Skipping {frame_sparsity}: no JSON files found")
                continue
            
            print(f"  Processing {frame_sparsity} ({len(json_files)} JSON files)...")
            process_folder(folder_path, output_file)
            print(f"  Results saved to {output_file}")

def main():
    # Get the directory where this script is located (results/)
    results_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Unified results generator")
    print(f"Base directory: {results_dir}")
    
    # Process both baseline and dynastride results
    baseline_eval_dir = os.path.join(results_dir, "baseline_results", "evaluation_results")
    dynastride_eval_dir = os.path.join(results_dir, "dynastride_results", "evaluation_results")
    
    # Process baseline results
    if os.path.isdir(baseline_eval_dir):
        process_evaluation_results(baseline_eval_dir)
    
    # Process dynastride results
    if os.path.isdir(dynastride_eval_dir):
        process_evaluation_results(dynastride_eval_dir)
    
    print(f"\n{'='*60}")
    print("All processing complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

