#!/bin/bash

# Check if exactly 1 argument was passed
if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder_number>"
    exit 1
fi

FOLDER_NUM=$1
BASE_PATH="/workspace/scene_captioner/data/YouCookII/YouCookII/raw_videos/training/$FOLDER_NUM"

# Number of parallel processes — adjust based on CPU cores and I/O
PARALLEL_JOBS=32

# Check if the folder exists
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: Folder $BASE_PATH does not exist."
    exit 1
fi

echo "Converting all PNGs to JPG in: $BASE_PATH using $PARALLEL_JOBS parallel jobs..."
find "$BASE_PATH" -type f -name "*.png" -print0 | \
xargs -0 -P $PARALLEL_JOBS -n 1 mogrify -format jpg -quality 85

echo "Deleting original PNGs in: $BASE_PATH ..."
find "$BASE_PATH" -type f -name "*.png" -delete

echo "✅ Conversion and cleanup complete for folder $FOLDER_NUM"
