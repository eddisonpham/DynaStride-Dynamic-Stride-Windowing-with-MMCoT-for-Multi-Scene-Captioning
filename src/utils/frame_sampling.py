"""Frame sampling utilities for video captioning."""
import glob
import os
from typing import List


def sample_frames(frame_folder: str, stride: int = 5, limit: int = 10) -> List[str]:
    """Sample frames from a folder with stride and limit.
    
    Args:
        frame_folder: Directory containing frame images
        stride: Stride for initial sampling (take every Nth frame)
        limit: Maximum number of frames to return
    
    Returns:
        List of frame file paths
    """
    jpgs = sorted(
        glob.glob(os.path.join(frame_folder, "*.jpg")) +
        glob.glob(os.path.join(frame_folder, "*.jpeg"))
    )
    if not jpgs:
        return []
    
    sampled = jpgs[::max(1, stride)]
    if len(sampled) > limit:
        idxs = [0]
        mids = max(0, limit - 2)
        if mids:
            step = max(1, (len(sampled) - 2) // mids)
            idxs += list(range(1, len(sampled) - 1, step))[:mids]
        idxs += [len(sampled) - 1]
        sampled = [sampled[i] for i in idxs]
    return sampled


def evenly_sample(items: List[str], k: int) -> List[str]:
    """Evenly sample k items from a list."""
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return items
    step = n / float(k)
    return [items[int(i * step)] for i in range(k)]

