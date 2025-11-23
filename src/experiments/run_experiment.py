"""Unified experiment runner for video captioning experiments."""
import os
import json
import time
import pathlib
import argparse
from typing import Dict
import sys
from dotenv import load_dotenv
ROOT_DIR=os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
print(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
load_dotenv(dotenv_path = os.path.join(ROOT_DIR, "src/experiments/.env") )
gptKey = os.getenv("OPEN_AI_API_KEY")
from src.utils.loader_utils import (
    get_data_paths,
    read_sample_file,
    get_ground_truth,
    find_scene_folders,
)
from src.utils.frame_sampling import sample_frames
from src.baselines.gpt_captioner import caption_with_openai
from src.baselines.videollama3_captioner import caption_with_videollama3


def get_output_filename(model: str, max_frames: int, round_num: int, paths: Dict[str, str]) -> str:
    """Generate output filename based on experiment parameters."""
    if model.lower() == "gpt":
        base_name = f"gpt4o-captions{max_frames}"
    elif model.lower() in ["videollama3"]:
        base_name = f"videollama3-captions{max_frames}"
    else:
        raise ValueError(f"Unknown model: {model}. Must be 'gpt' or 'videollama3'")
    
    if round_num > 1:
        filename = f"{base_name}-round{round_num}.json"
    else:
        filename = f"{base_name}.json"
    
    return os.path.join(paths["outputDir"], filename)


def run_experiment(
    model: str,
    max_frames: int,
    round_num: int = 1,
    stride: int = None
):
    """Run a captioning experiment."""
    if model.lower() not in ["gpt", "videollama3"]:
        raise ValueError(f"Unknown model: {model}. Must be 'gpt' or 'videollama3'")
    
    if stride is None:
        stride = 5 if model.lower() == "gpt" else 8
    
    if model.lower() == "gpt":
        if gptKey is None:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    
    paths = get_data_paths()
    output_file = get_output_filename(model, max_frames, round_num, paths)
    
    print(f"[start] model={model} max_frames={max_frames} round={round_num} baseDir={paths['baseDir']}", flush=True)
    print(f"[start] output={output_file}", flush=True)
    
    print("[stage] loading ground truth...", flush=True)
    gt = get_ground_truth(paths["gtFile"])
    print(f"[stage] ground truth loaded: {len(gt)} videos", flush=True)
    
    lines = read_sample_file(paths["sampleFile"])
    print(f"[stage] samples loaded: {len(lines)} entries from {paths['sampleFile']}", flush=True)
    
    merged: Dict[str, Dict[str, Dict[str, str]]] = {}
    processed = 0
    started = time.time()
    req_delay = float(os.environ.get("OPENAI_REQ_DELAY", "0.7")) if model.lower() == "gpt" else 0.1
    
    for rel in lines:
        video_id = pathlib.Path(rel).parts[-1]
        video_root = os.path.join(paths["baseDir"], rel)
        print(f"[video] {video_id} scanning scenes in {video_root}", flush=True)
        
        scene_folders = find_scene_folders(video_root)
        print(f"[video] {video_id} found {len(scene_folders)} scene folders", flush=True)
        if not scene_folders:
            continue
        
        merged.setdefault(video_id, {})
        
        for idx, scene_path in enumerate(scene_folders):
            t0 = time.time()
            folder_name = os.path.basename(scene_path)
            try:
                scene_index = int(folder_name) if folder_name.isdigit() else idx
            except ValueError:
                scene_index = idx
            
            print(f"[scene] {video_id}/{scene_index} sampling frames from {scene_path}", flush=True)
            frame_paths = sample_frames(scene_path, stride=stride, limit=max_frames)
            print(f"[scene] {video_id}/{scene_index} sampled {len(frame_paths)} frames", flush=True)
            
            if not frame_paths:
                continue
            
            # Generate caption
            try:
                if model.lower() == "gpt":
                    pred_raw = caption_with_openai(frame_paths, max_frames=max_frames)
                else:
                    pred_raw = caption_with_videollama3(frame_paths)
                print(f"[scene] {video_id}/{scene_index} caption received", flush=True)
            except Exception as e:
                print(f"[scene] {video_id}/{scene_index} failed: {e}", flush=True)
                if model.lower() in ["llama3", "videollama3"]:
                    time.sleep(0.25)
                    try:
                        pred_raw = caption_with_videollama3(frame_paths)
                        print(f"[scene] {video_id}/{scene_index} caption received (retry)", flush=True)
                    except Exception as e2:
                        print(f"[scene] {video_id}/{scene_index} retry failed: {e2}", flush=True)
                        pred_raw = ""
                else:
                    pred_raw = ""
            
            gt_sentence = gt.get(video_id, {}).get(scene_index, "")
            merged[video_id][str(scene_index)] = {
                "ground_truth": gt_sentence,
                "predicted": pred_raw
            }
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            
            dt = time.time() - t0
            processed += 1
            elapsed = time.time() - started
            avg = elapsed / processed if processed else 0.0
            print(f"[scene] {video_id}/{scene_index} done in {dt:.1f}s | total={processed} | avg={avg:.1f}s/scene | wrote {output_file}", flush=True)
            
            time.sleep(req_delay)
    
    total = time.time() - started
    print(f"[done] wrote {output_file} | scenes={processed} | total={total:.1f}s | avg={(total/processed) if processed else 0.0:.1f}s/scene", flush=True)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run video captioning experiment")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt", "videollama3"],
        default="gpt",
        help="Model to use: 'gpt' or 'videollama3'"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        choices=[5, 10, 20, 40],
        help="Maximum frames to sample per scene (every K-th frame sparsity)"
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Round/seed number (default: 1)"
    )
    gptStride = 5
    llamaStride = 8
    
    args = parser.parse_args()
    run_experiment(
        model=args.model,
        max_frames=args.max_frames,
        round_num=args.round,
        stride= gptStride if args.model == "gpt" else llamaStride,
    )


if __name__ == "__main__":
    main()

