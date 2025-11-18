"""
Extracts and deletes the following archives if present:
- scripts.tar.gz
- splits.tar.gz
- youcookii_annotations_trainval.tar.gz
"""

import os
import tarfile

TAR_FILES = [
    "scripts.tar.gz",
    "splits.tar.gz",
    "youcookii_annotations_trainval.tar.gz"
]

def extract_and_delete_tar_gz(rel_dir, tar_name):
    """Extract a tar.gz archive at the given directory (relative) and delete it."""
    try:
        tar_path = os.path.join(rel_dir, tar_name)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=rel_dir)
        os.remove(tar_path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to extract or remove: {tar_name}") from e

if __name__ == "__main__":
    rel_dir = os.path.dirname(__file__)
    for tar_name in TAR_FILES:
        extract_and_delete_tar_gz(rel_dir, tar_name)
