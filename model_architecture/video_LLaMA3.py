#!/usr/bin/env python3
"""
video_LLaMA3.py
-----------
A pipeline for setting up the VideoLLaMA environment, generating image captions,
extracting frames from videos, and producing coherent video-level captions.
"""

import os
import subprocess
from typing import List
import cv2  # For frame extraction
from transformers import AutoModelForCausalLM, AutoProcessor
import torch


def setup(repo_url: str = "https://github.com/DAMO-NLP-SG/VideoLLaMA3.git", repo_dir: str = "VideoLLaMA3"):
    """
    Sets up the environment by cloning the model repository and installing dependencies.
    Only installs requirements if needed, and defaults to known dependencies if 
    requirements.txt is missing.
    """
    # Checking if the repo already exists or not
    if not os.path.exists(repo_dir):
        print(f"[INFO] Cloning repository {repo_url}...")
        subprocess.run(["git", "clone", repo_url], check=True)
    else:
        print(f"[INFO] Repository '{repo_dir}' already exists. Skipping clone.")

    # Checks if a requirements.txt is present in the folder
    #   - may have to change this if requirements.txt is in another folder/has another path
    req_path = "requirements.txt"
    if os.path.exists(req_path):
        print(f"[INFO] Found requirements.txt at {req_path}. Installing...")
        subprocess.run(["pip", "install", "-r", req_path], check=True)
    else:
        # the default dependencies mentioned here are the dependencies commented out below
        # this can be sorted out further once we have a unified requirements.txt
        print(f"[INFO] No requirements.txt found. Using default dependencies.")

# Load model and processor globally to avoid reloading for each call
# We need to look into changing this for our unified requirements.txt (this should be good for now)
!echo ">>> Uninstalling potential conflicts..."
!pip uninstall -y torch torchvision torchaudio flash-attn transformers accelerate nvidia-pyindex nvidia-pip || true

!echo ">>> Upgrading pip..."
!pip install --upgrade pip

!echo ">>> Installing PyTorch 2.3.1 + CUDA 12.1..."
!pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

!echo ">>> Installing latest Transformers + Accelerate..."
!pip install transformers==4.46.3 accelerate==1.0.1

!echo ">>> Installing other dependencies..."
!pip install decord ffmpeg-python imageio opencv-python

!echo ">>> Setting CUDA environment variables..."
!export CUDA_HOME=/usr/local/cuda-12.1
!export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

!echo ">>> Setup Complete!

# we run the setup and make the model global
setup()
print("[INFO] Loading VideoLLaMA model...")
MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA3-7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    offload_folder="offload"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("[INFO] Model loaded successfully.")


def generate_caption(images: List[str], prompt: str) -> str:
    """
    Generates a caption for one or more images given a text prompt.

    :param images: List of image file paths.
    :param prompt: Text prompt to guide the caption generation.
    :return: Generated caption as a string.
    """
    print(f"[INFO] Generating caption for {len(images)} image(s)...")
    conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            *[{"type": "image", "image": {"image_path": img}} for img in images],
            {"type": "text", "text": prompt}
        ]
    },
    ]
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
      inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=64)
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return caption


def extract_frames(video_path: str, output_dir: str, frame_interval: int = 120) -> List[str]:
    """
    Extracts frames from a video at a specified interval.

    :param video_path: Path to the input video.
    :param output_dir: Directory where extracted frames will be saved.
    :param frame_interval: Save every `frame_interval` frames (default: 120).
    :return: List of file paths to extracted frames.
    """
    print(f"[INFO] Extracting frames from {video_path} every {frame_interval} frames...")

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, 0
    frame_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_paths.append(frame_filename)
            saved += 1
        count += 1

    cap.release()
    print(f"[INFO] Extracted {len(frame_paths)} frames.")
    return frame_paths


def generate_video_caption(frames: List[str]) -> str:
    """
    Combines frame-wise captions into a single, coherent video-level caption.

    :param frames: List of frame file paths.
    :return: Final video caption string.
    """
    print("[INFO] Generating video-level caption...")
    frame_captions = [generate_caption([frame], "You are given a single frame extracted from a video. Generate a detailed and contextually rich caption that describes not only what is visually present (objects, people, setting, actions, emotions) but also infers the likely context of the scene based on visual cues such as lighting, expressions, clothing, or background elements. If the frame seems to be part of an ongoing action or event, include your best guess at what is happening before and after this moment, using natural language to provide a cohesive narrative. Be concise but descriptive, and focus on making the caption feel like a natural description someone might give while watching the video. Limit 64 tokens") for frame in frames]

    # Use the model to merge frame captions into a coherent summary
    merged_prompt = " ".join(frame_captions)
    summary_input = f"Merge these captions into one continuous, coherent, and natural-sounding description of the entire video. Avoid repeating the same details unless necessary, ensure smooth transitions between actions, and infer the overall context or story from the captions. If possible, describe the progression of events as if narrating the video from start to finish: {merged_prompt}"
    final_caption = generate_caption([], summary_input)
    return final_caption


if __name__ == "__main__":
    # Example usage (comment or remove in production)
    print("[INFO] Example pipeline execution (modify paths before running).")
    #setup()
    caption = generate_caption(["/content/VideoLLaMA3/assets/logo.png"], "Describe the image")
    # frames = extract_frames("/content/VideoLLaMA3/assets/cat_and_chicken.mp4", "frames_output", frame_interval=120)
    # caption = generate_video_caption(frames)
    print("Final Video Caption:", caption)
