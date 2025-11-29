import torch
import re
import gc
from typing import List, Tuple, Optional, Union
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

_model_cache = {}


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_id: str = "zai-org/GLM-4.5V", 
    device: Optional[torch.device] = None, 
    token: bool = True
) -> Tuple[AutoProcessor, AutoModelForImageTextToText]:
    """
    Load a vision-language model with caching.
    
    Args:
        model_id: HuggingFace model identifier
        device: Target device (if None, uses get_device())
        token: Whether to use authentication token
        
    Returns:
        Tuple of (processor, model)
    """
    global _model_cache
    
    if device is None:
        device = get_device()
    
    # Return cached model if available
    cache_key = f"{model_id}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        processor = AutoProcessor.from_pretrained(model_id, token=token)
        model = AutoModelForImageTextToText.from_pretrained(model_id, token=token).to(device)
        
        _model_cache[cache_key] = (processor, model)
        return processor, model
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_id}: {e}")

def load_frame(frame_path: str) -> Image.Image:
    """
    Load a frame image from file path.
    
    Args:
        frame_path: Path to the frame image
        
    Returns:
        PIL Image in RGB format
    """
    try:
        return Image.open(frame_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to load frame {frame_path}: {e}")

def concat_images(
    images: List[Union[str, Image.Image]], 
    horizontal: bool = True, 
    max_side: int = 768
) -> Image.Image:
    """
    Concatenate a list of images horizontally or vertically.
    Resizes images proportionally if larger than max_side.
    
    Args:
        images: List of image paths or PIL Image objects
        horizontal: If True, concatenate horizontally; else vertically
        max_side: Maximum side length for resizing
        
    Returns:
        Concatenated PIL Image
    """
    if not images:
        raise ValueError("Empty images list provided")
    
    pil_images = []
    for img in images:
        try:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")

            w, h = img.size
            scale = max(w, h) / float(max_side)
            if scale > 1.0:
                img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)

            pil_images.append(img)
        except Exception as e:
            raise RuntimeError(f"Failed to process image: {e}")

    if not pil_images:
        raise ValueError("No valid images to concatenate")

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

def generate_conclusion(
    processor: AutoProcessor, 
    model: AutoModelForImageTextToText, 
    inputs: dict, 
    max_new_tokens: int = 200
) -> Optional[str]:
    """
    Generate text conclusion from processed inputs.
    Decodes only the new tokens without including the input.
    
    Args:
        processor: Model processor
        model: Vision-language model
        inputs: Processed input tensors
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Extracted conclusion text or None
    """
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                do_sample=True
            )
        
        # Decode only generated tokens
        generated_text = processor.batch_decode(
            outputs.sequences[:, inputs["input_ids"].shape[1]:],  # slice after prompt
            skip_special_tokens=True
        )[0]

        return extract_conclusion(generated_text)
    except Exception as e:
        return None

def analyze_sequence_by_indexes(
    folder: str,
    frame_indexes: List[str],
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    device: torch.device,
    max_new_tokens: int = 200
) -> Optional[str]:
    """
    Analyze a sequence of frames using the vision-language model.
    
    Args:
        folder: Directory containing frame images
        frame_indexes: List of frame index strings (e.g., ['0000', '0010'])
        processor: Model processor
        model: Vision-language model
        device: Target device
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Extracted conclusion text or None
    """
    if not frame_indexes:
        return None
    
    try:
        # Load and concatenate frames
        frame_paths = [f"{folder}/frame_{idx}.jpg" for idx in frame_indexes]
        images_to_concat = [load_frame(path) for path in frame_paths]
        multi_image_input = concat_images(images_to_concat, horizontal=True)

        # Create messages and process inputs
        messages = create_message(multi_image_input, num_images=len(images_to_concat))
        inputs = process_inputs(processor, messages, device)

        # Generate conclusion
        conclusion = generate_conclusion(processor, model, inputs, max_new_tokens=max_new_tokens)

        # Cleanup
        del images_to_concat, multi_image_input, inputs
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return conclusion
    except Exception as e:
        # Cleanup on error
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return None