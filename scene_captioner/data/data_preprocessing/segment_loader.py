import os
import pickle
import numpy as np
import cv2
import re
import shutil
import sys
from pathlib import Path
from collections import defaultdict
from decord import VideoReader, cpu

"""
To run this file, include an argument to specify which split you would like to segment into frames
By default, this is set to the training split
"""

set_type = "training"
if len(sys.argv) == 2:
    set_type = sys.argv[1]
    if set_type != "training" and set_type != "validation" and set_type != "testing":
        print(f" Please enter a valid data set split, {set_type} is not valid")
        sys.exit(1)

class YouCook2SegmentLoader:
    def __init__(self, videos_dir, refs_pkl_path, img_size=384):
        """
        :param: videos_dir (str or Path): folder containing YouCook2 .mp4 videos
        :param: refs_pkl_path (str or Path): pkl file from preprocessing step
        :paramimg_size (int): frame resize dimension (square)
        """
        self.videos_dir = Path(videos_dir)
        self.img_size = img_size

        with open(refs_pkl_path, "rb") as f:
            self.references = pickle.load(f)

        print(f"Loaded {len(self.references)} segments from {refs_pkl_path}")

    # creates a path to a video using the ID
    def _get_video_path(self, video_id):
        video_path = self.videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        return video_path

    # this will deal with np.arrays (i'm keeping this here for now, in case we want to revert)
    def load_segment_frames(self, seg_id, video_path, with_caption=True):
        """
        Load the exact frames for a given segment ID.
        
        :param - seg_id (str): e.g., 'VID123_0'
        :param - with_caption (bool): return associated GT caption if True

        :return - frames: np.ndarray of shape (T, H, W, C)
        :return - caption: (optional) str, GT caption
        """
        meta_list = self.references.get(seg_id, None)
        if meta_list is None:
            raise ValueError(f"No metadata for segment {seg_id}")

        meta = meta_list[0]   # first entry (only one in most cases)
        video_id = seg_id.rsplit("_", 1)[0]
        start_f = meta['startFrame']
        end_f   = meta['endFrame']

        #video_path = self._get_video_path(video_id)
        vr = VideoReader(str(video_path), ctx=cpu(0))
        vlen = len(vr)
        if end_f >= vlen:
            raise ValueError(f"End frame {end_f} >= video length {vlen} in {video_id}")

        indices = list(range(start_f, end_f + 1))
        frames = vr.get_batch(indices).asnumpy()

        if self.img_size:
            frames = np.stack([
                cv2.resize(frame, (self.img_size, self.img_size)) 
                for frame in frames
            ])

        if with_caption:
            return frames, meta['caption']
        return frames

    def load_frames(self, seg_id, with_caption, video_path, save_dir_root):
        meta_list = self.references.get(seg_id, None)
        if meta_list is None:
            raise ValueError(f"No metadata for the segment {seg_id}")
        meta = meta_list[0]
        video_id = seg_id.rsplit("_", 1)[0]
        start_f = meta['startFrame']
        end_f = meta['endFrame']

        # if the number of frames in the data exceeds the actual number in the video
        #    - this could be a problem with the way fps is calculated
        vr = VideoReader(str(video_path), ctx=cpu(0))
        vlen = len(vr)
        if end_f >= vlen:
            #raise ValueError(f"End frame {end_f} >= video length {vlen} in {video_id}")
            print(f"End frame {end_f} >= video length {vlen} in {video_id}")
            end_f = vlen

        indices = list(range(start_f, end_f + 1))
        frames = vr.get_batch(indices).asnumpy()

        if self.img_size:
            frames = [cv2.resize(f, (self.img_size, self.img_size)) for f in frames]
        else:
            frames = list(frames)
        
        # creating a folder for saving the segment frames
        segment_folder = Path(save_dir_root) / f"{video_id}/{seg_id}_frames"
        segment_folder.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, frame in enumerate(frames):
            filename = segment_folder / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(filename), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            saved_paths.append(str(filename))
        if with_caption == True:
            return saved_paths, meta['caption']
        else:
            return saved_paths
        
    def get_all_segment_ids(self):
        return list(self.references.keys())


def count_image_files(folder_path, extensions=('.png', '.jpg')):
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                count += 1
    return count

# ========== Example usage ==========
if __name__ == "__main__":
    videos_path = f"/workspace/scene_captioner/data/YouCookII/YouCookII/raw_videos/{set_type}/"
    refs_path = "/workspace/scene_captioner/data/YouCookII/YouCookII/saved_references/"
    if set_type == "testing":
        refs_path += "youcook2_test_refs.pkl"
    elif set_type == "validation":
        refs_path += "youcook2_val_refs.pkl"
    else:
        refs_path += "youcook2_train_refs.pkl"
    loader = YouCook2SegmentLoader(videos_path, refs_path, img_size=384)

    # going through all of the videos we've downloaded so far
    segment_ids = loader.get_all_segment_ids()
    vid_to_segments = defaultdict(list)
    for seg_id in segment_ids:
        vid = seg_id.rsplit("_", 1)[0]
        vid_to_segments[vid].append(seg_id)

    # walk through all of the subfolders
    vid_count = 0
    for root, dirs, files in os.walk(videos_path):
        for file in files:
            if file.endswith(".mp4"):
                video_id = file[:-4]  # strip ".mp4"
                video_path = os.path.join(root, file)
                print(video_path)

                # getting the saving root directory for this video
                match = re.match(r"(.*/\d+/).*", video_path)
                save_root_dir = None
                if match: 
                    save_root_dir = match.group(1)

                # checking if the folder for the video already exists
                vid_folder_path = f"{save_root_dir}/{video_id}"
                if os.path.exists(vid_folder_path):
                    num_files = count_image_files(vid_folder_path)
                    if num_files < 5:  # or use < 10 depending on your threshold
                        shutil.rmtree(vid_folder_path)  # deletes the entire directory and its contents
                        print(f"Deleted {vid_folder_path} since it contained only {num_files} image files.")
                    else:
                        # Continue processing if enough files are present
                        vid_count += 1
                        continue
                
                print(f"Processing video: {video_id}")
                # Find all segments corresponding to this video
                segments = vid_to_segments.get(video_id, [])
                if not segments:
                    print(f"  No segments found for {video_id}")
                    continue
    
                # Load frames and captions for each segment
                success = 0
                for seg_id in segments:
                    try:
                        if set_type == "testing":
                            with_caption = False
                            frames = loader.load_frames(seg_id, with_caption, video_path, save_root_dir)
                            print(f"  Segment {seg_id}: # of frames = {len(frames)}")
                        else:
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