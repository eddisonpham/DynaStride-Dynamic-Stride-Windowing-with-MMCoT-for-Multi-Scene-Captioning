"""YouCook2 segment loader for frame extraction."""
import os
import pickle
import numpy as np
import cv2
import re
import shutil
import sys
from pathlib import Path
from collections import defaultdict


def _read_frames_cv2(video_path, frame_indices):
    """Read specific frames from a video using OpenCV."""
    if not frame_indices:
        return []
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    sorted_indices = sorted(frame_indices)
    
    is_consecutive = (len(sorted_indices) == 1 or 
                     (sorted_indices[-1] - sorted_indices[0] == len(sorted_indices) - 1))
    
    if is_consecutive:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sorted_indices[0])
        for _ in sorted_indices:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    else:
        for idx in sorted_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames


class YouCook2SegmentLoader:
    """Loader for YouCook2 video segments."""
    
    def __init__(self, videos_dir, refs_path, img_size=384):
        self.videos_dir = Path(videos_dir)
        self.img_size = img_size

        with open(refs_path, "rb") as f:
            self.references = pickle.load(f)

        print(f"Loaded {len(self.references)} segments from {refs_path}")

    def _get_video_path(self, video_id):
        """Create path to a video using the ID."""
        video_path = self.videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        return video_path

    def load_segment_frames(self, seg_id, video_path, with_caption=True):
        """Load the exact frames for a given segment ID."""
        meta_list = self.references.get(seg_id, None)
        if meta_list is None:
            raise ValueError(f"No metadata for segment {seg_id}")

        meta = meta_list[0]
        video_id = seg_id.rsplit("_", 1)[0]
        start_f = meta['startFrame']
        end_f = meta['endFrame']

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if end_f >= vlen:
            raise ValueError(f"End frame {end_f} >= video length {vlen} in {video_id}")

        indices = list(range(start_f, end_f + 1))
        frames = _read_frames_cv2(video_path, indices)

        if self.img_size:
            frames = np.stack([
                cv2.resize(frame, (self.img_size, self.img_size)) 
                for frame in frames
            ])
        else:
            frames = np.stack(frames)

        if with_caption:
            return frames, meta['caption']
        return frames

    def load_frames(self, seg_id, with_caption, video_path, save_dir_root):
        """Load frames and optionally save them to disk."""
        meta_list = self.references.get(seg_id, None)
        if meta_list is None:
            raise ValueError(f"No metadata for the segment {seg_id}")
        meta = meta_list[0]
        video_id = seg_id.rsplit("_", 1)[0]
        start_f = meta['startFrame']
        end_f = meta['endFrame']

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if end_f >= vlen:
            print(f"End frame {end_f} >= video length {vlen} in {video_id}")
            end_f = vlen - 1

        indices = list(range(start_f, end_f + 1))
        frames = _read_frames_cv2(video_path, indices)

        if self.img_size:
            frames = [cv2.resize(f, (self.img_size, self.img_size)) for f in frames]
        else:
            frames = list(frames)
        
        segment_folder = Path(save_dir_root) / f"{video_id}/{seg_id}_frames"
        segment_folder.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, frame in enumerate(frames):
            filename = segment_folder / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(filename), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            saved_paths.append(str(filename))
        if with_caption:
            return saved_paths, meta['caption']
        else:
            return saved_paths
        
    def get_all_segment_ids(self):
        """Get all segment IDs."""
        return list(self.references.keys())


def count_image_files(folder_path, extensions=('.png', '.jpg')):
    """Count image files in a directory."""
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                count += 1
    return count


def process_video_segments():
    """Process video segments for a given dataset split."""
    WORKSPACE = os.path.abspath(os.path.join(os.getcwd(), ".."))
    SCENE_CAPTIONER_DATA = os.path.join(WORKSPACE, "YouCookII")
    
    videos_path = os.path.join(SCENE_CAPTIONER_DATA, f"raw_videos/validation/")
    refs_path = os.path.join(SCENE_CAPTIONER_DATA, "saved_references", f"youcook2_validation_refs.pkl")
    
    loader = YouCook2SegmentLoader(videos_path, refs_path, img_size=384)

    segment_ids = loader.get_all_segment_ids()
    vid_to_segments = defaultdict(list)
    for seg_id in segment_ids:
        vid = seg_id.rsplit("_", 1)[0]
        vid_to_segments[vid].append(seg_id)

    vid_count = 0
    for root, dirs, files in os.walk(videos_path):
        for file in files:
            if file.endswith(".mp4"):
                video_id = file[:-4]
                video_path = os.path.join(root, file)
                print(video_path)

                match = re.match(r"(.*/\d+/).*", video_path)
                save_root_dir = None
                if match: 
                    save_root_dir = match.group(1)

                vid_folder_path = f"{save_root_dir}/{video_id}"
                if os.path.exists(vid_folder_path):
                    num_files = count_image_files(vid_folder_path)
                    if num_files < 5:
                        shutil.rmtree(vid_folder_path)
                        print(f"Deleted {vid_folder_path} since it contained only {num_files} image files.")
                    else:
                        vid_count += 1
                        continue
                
                print(f"Processing video: {video_id}")
                segments = vid_to_segments.get(video_id, [])
                if not segments:
                    print(f"  No segments found for {video_id}")
                    continue
    
                success = 0
                for seg_id in segments:
                    try:
                        with_caption = True
                        frames, caption = loader.load_frames(seg_id, with_caption, video_path, save_root_dir)
                        print(f"  Segment {seg_id}: # of frames = {len(frames)}, caption = {caption}")
                        success = 0
                    except Exception as e:
                        success = 1
                        print(f"  Error loading segment {seg_id}: {e}")

                if success == 0:
                    vid_count += 1

                if vid_count % 10 == 0:
                    print(f"\nVideos Done so Far: {vid_count}\n")


if __name__ == "__main__":
    process_video_segments()

