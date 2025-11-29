"""VideoLLaMA3 model wrapper for video captioning."""
import os
import re
import time
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

PROMPT_VIDEOLLAMA = (
    "You are given multiple frames from a short cooking clip, in chronological order. "
    "Write ONE concise sentence that is both descriptive, short, and instructional. "
    "Use an imperative tone, as if giving instructions for cooking or performing a task. "
    "Your response MUST be enclosed between <ANSWER> and </ANSWER>, containing ONLY the final instruction sentence."
)

STRIDE = 8
VIDEOLLAMA_NAME = "DAMO-NLP-SG/VideoLLaMA3-7B"

_model = None
_processor = None
_device = None


def _initialize_model():
    """Initialize VideoLLaMA3 model and processor."""
    global _model, _processor, _device
    
    if _model is not None:
        return _model, _processor, _device
    
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    offload_folder = "offload"
    os.makedirs(offload_folder, exist_ok=True)
    
    print(f"[start] device={_device} | model={VIDEOLLAMA_NAME}", flush=True)
    t0_load = time.time()
    print("[stage] loading VideoLLaMA3 weights...", flush=True)
    
    _model = AutoModelForCausalLM.from_pretrained(
        VIDEOLLAMA_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=(torch.float16 if _device == "cuda" else torch.float32),
        low_cpu_mem_usage=True,
        offload_folder=offload_folder
    )
    _processor = AutoProcessor.from_pretrained(VIDEOLLAMA_NAME, trust_remote_code=True)
    
    print(f"[stage] model ready in {time.time()-t0_load:.1f}s", flush=True)
    return _model, _processor, _device


def caption_with_videollama3(paths: List[str]) -> str:
    """Generate caption from frame paths using VideoLLaMA3."""
    model, processor, device = _initialize_model()
    
    print(f"[infer] preparing {len(paths)} frames", flush=True)
    content = [{"type": "image", "image": {"image_path": p}} for p in paths]
    content.append({"type": "text", "text": PROMPT_VIDEOLLAMA})
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]
    tprep = time.time()
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    tensor_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            tensor_inputs[k] = v.to(device)
        else:
            tensor_inputs[k] = v
    if "pixel_values" in tensor_inputs and tensor_inputs["pixel_values"].dtype != torch.float16 and device == "cuda":
        tensor_inputs["pixel_values"] = tensor_inputs["pixel_values"].to(torch.float16)
    print(f"[infer] inputs ready in {time.time()-tprep:.1f}s; generating...", flush=True)
    tgen = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **tensor_inputs,
            max_new_tokens=60,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f"[infer] generation done in {time.time()-tgen:.1f}s", flush=True)
    m = re.search(r"<ANSWER>(.*?)</ANSWER>", text, flags=re.IGNORECASE | re.DOTALL)
    pred = (m.group(1) if m else text).strip()
    print(f"[infer] caption: {pred}", flush=True)
    return pred

