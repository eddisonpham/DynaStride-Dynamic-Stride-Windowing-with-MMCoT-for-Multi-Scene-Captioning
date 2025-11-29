"""Image processing utilities for video captioning."""
import io
import base64
import os
import sys
from typing import List
from PIL import Image

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.frame_sampling import evenly_sample


def b64_size_bytes(data_url: str) -> int:
    """Calculate base64 data size in bytes."""
    b64 = data_url.split(",", 1)[1]
    return (len(b64) * 3) // 4


def make_horizontal_strip_data_url(
    paths: List[str], 
    max_height: int, 
    max_frames: int, 
    gap: int, 
    quality: int
) -> str:
    """Create a horizontal strip image from frame paths and return as base64 data URL."""
    print(f"[strip] building strip: frames={max_frames} height={max_height} gap={gap} q={quality}", flush=True)
    chosen = evenly_sample(paths, min(max_frames, len(paths)))
    frames = []
    for p in chosen:
        try:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            if h != max_height:
                new_w = max(1, int(w * (max_height / float(h))))
                img = img.resize((new_w, max_height), Image.LANCZOS)
            frames.append(img)
        except Exception as e:
            print(f"[strip] skip frame {p}: {e}", flush=True)
            continue
    if not frames:
        raise RuntimeError("no frames available to build strip")
    total_w = sum(im.width for im in frames) + gap * (len(frames) - 1)
    strip = Image.new("RGB", (total_w, max_height), (255, 255, 255))
    x = 0
    for i, im in enumerate(frames):
        strip.paste(im, (x, 0))
        x += im.width + (gap if i < len(frames) - 1 else 0)
    bio = io.BytesIO()
    strip.save(bio, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
    url = f"data:image/jpeg;base64,{b64}"
    print(f"[strip] built strip {total_w}x{max_height}, payload≈{b64_size_bytes(url)/1024:.1f}KB", flush=True)
    return url

