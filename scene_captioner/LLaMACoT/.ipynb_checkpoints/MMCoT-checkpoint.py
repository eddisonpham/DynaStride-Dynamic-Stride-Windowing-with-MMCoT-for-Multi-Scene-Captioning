# filename: llama_temporal.py

# -------------------------------
# Step 1: Set environment variables
# -------------------------------
import os
os.environ['HF_HOME'] = '/workspace/hf'
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.environ['HF_HOME'], "transformers")
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.environ['HF_HOME'], "datasets")

# -------------------------------
# Step 2: Install dependencies
# -------------------------------
import subprocess
import sys

def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "transformers"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "Pillow"])

install_packages()

# -------------------------------
# Step 3: Import dependencies
# -------------------------------
import torch
import re
from PIL import Image, ImageOps
from transformers import AutoProcessor, AutoModelForImageTextToText
import gc
from typing import List

# -------------------------------
# Utility functions
# -------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Model & Processor
# -------------------------------
_model_cache = {}

def load_model(model_id="zai-org/GLM-4.5V", device=None, token=True):
    global _model_cache
    if model_id in _model_cache:
        return _model_cache[model_id]

    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = AutoModelForImageTextToText.from_pretrained(model_id, token=token).to(device)

    _model_cache[model_id] = (processor, model)
    return processor, model

# -------------------------------
# Image Utilities
# -------------------------------
def load_frame(frame_path):
    return Image.open(frame_path).convert("RGB")

def concat_images(images, horizontal=True, max_side=768):
    """
    Concatenate a list of images horizontally or vertically.
    Resizes images proportionally if larger than max_side.
    Returns a single PIL.Image object.
    """
    pil_images = []

    # Convert all items to PIL.Image if they are paths
    for img in images:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")

        # Resize proportionally if too large
        w, h = img.size
        scale = max(w, h) / float(max_side)
        if scale > 1.0:
            img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)

        pil_images.append(img)

    # Compute final canvas size
    widths, heights = zip(*(i.size for i in pil_images))
    if horizontal:
        total_width = sum(widths)
        max_height = max(heights)
        new_img = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in pil_images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width
    else:
        total_height = sum(heights)
        max_width = max(widths)
        new_img = Image.new("RGB", (max_width, total_height))
        y_offset = 0
        for img in pil_images:
            new_img.paste(img, (0, y_offset))
            y_offset += img.height

    return new_img

# -------------------------------
# HuggingFace Chat Utilities
# -------------------------------
def create_message(image, num_images=5):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f"USER: {'<image> ' * num_images}\n"
                        "These images show a sequence of events from left to right.\n"
                        "Task:\n"
                        "1. Carefully reason about the sequence of actions in the images (think step by step internally).\n"
                        "2. Then, produce the following outputs separated by a '|' character:\n"
                        "   - Output 1: Description of the exact action being performed throughout the sequence.\n"
                        "   - Output 2: List of as many objects involved in the sequence of images.\n"
                        "3. Do NOT show your internal reasoning or extra captions—only the final two sentences.\n"
                        "4. Keep it short, clear, and concise.\n\n"
                        "Output format:\n"
                        "<CONCLUSION>Output 1 | Output 2</CONCLUSION>\n\n"
                        "Focus on the full temporal progression of the sequence, using internal reasoning to understand the events."
                    )
                }
            ],
        }
    ]




def process_inputs(processor, messages, device):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,  # avoid extra prompt tokens
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move all tensors to device and correct dtype
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if "pixel_values" in k:
                inputs[k] = v.to(device).to(torch.float16)
            else:
                inputs[k] = v.to(device).long()
    return inputs

def extract_conclusion(text):
    match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip().replace("\n", " ")
    return None

def generate_conclusion(processor, model, inputs, max_new_tokens=50):
    """
    Generate text, decoding only the new tokens without including the input.
    """
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            return_dict_in_generate=True,
            output_scores=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            do_sample=True
        )
    
    # decode only generated tokens
    generated_text = processor.batch_decode(
        outputs.sequences[:, inputs["input_ids"].shape[1]:],  # slice after prompt
        skip_special_tokens=True
    )[0]

    print(generated_text)

    return extract_conclusion(generated_text)

# -------------------------------
# High-level function
# -------------------------------
def analyze_sequence_by_indexes(folder, frame_indexes, processor, model, device, max_new_tokens=200):
    """
    Analyze a sequence of frames using the model.
    """
    images_to_concat = [load_frame(f"{folder}/frame_{idx}.jpg") for idx in frame_indexes]
    multi_image_input = concat_images(images_to_concat, horizontal=True)

    messages = create_message(multi_image_input, num_images=len(images_to_concat))
    inputs = process_inputs(processor, messages, device)

    conclusion = generate_conclusion(processor, model, inputs, max_new_tokens=max_new_tokens)

    # Free memory
    del images_to_concat, multi_image_input, inputs
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return conclusion

install_packages()



# -------------------------------
# Example run using specific frame indexes
# -------------------------------
# if __name__ == "__main__":
#     hf_token = input("Paste your HuggingFace token: ").strip()
#     setup_environment(hf_token=hf_token)
#     device = get_device()
    
#     folder_path = "/workspace/scene_captioner/data/YouCookII/YouCookII/raw_videos/training/101/2Ihlw5FFrx4/2Ihlw5FFrx4_0_frames"
#     frame_indexes = ["0000", "0001", "0002", "0003", "0004", "0005"]
    
#     conclusion = analyze_sequence_by_indexes(folder_path, frame_indexes, device=device)
#     print("CONCLUSION:", conclusion)
