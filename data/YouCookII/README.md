# YouCookII Dataset Setup

This folder contains the YouCookII dataset files needed for the project. The dataset files were originally distributed as compressed tar.gz archives and need to be extracted before use.

## Setup Instructions

### Step 1: Download Required Files

1. Visit the [YouCook2 Download Page](http://youcook2.eecs.umich.edu/download)

2. Download the following files:
   - **Annotations (Train+Val)**: `youcookii_annotations_trainval.tar.gz`
   - **Splits (Train+Val+Test)**: `splits.tar.gz`
   - **Scripts**: `scripts.tar.gz`

### Step 2: Place Files in YouCookII Folder

Drag and drop the downloaded `.tar.gz` files into this `YouCookII` folder:
```
data/YouCookII/
├── scripts.tar.gz
├── splits.tar.gz
├── youcookii_annotations_trainval.tar.gz
```

### Step 3: Extract the Archives

Run the extraction script from the project root:

```bash
python data/YouCookII/extract_tar.py
```

This script will:
- Extract `scripts.tar.gz` → creates `scripts/` folder
- Extract `splits.tar.gz` → creates `splits/` folder  
- Extract `youcookii_annotations_trainval.tar.gz` → creates `youcookii_annotations_trainval.json`
- Automatically delete the `.tar.gz` files after extraction

```bash
tar -xzf data/YouCookII/youcookii_annotations_test_segments_only.tar.gz -C data/YouCookII/
```

## Expected Folder Structure

After extraction, your `YouCookII` folder should contain:

```
data/YouCookII/
├── scripts/
│   └── download_youcookii_videos.py
├── splits/
│   ├── train_list.txt
│   ├── val_list.txt
│   ├── test_list.txt
│   └── *_duration_totalframe.csv
├── youcookii_annotations_trainval.json
├── extract_tar.py
└── README.md
```

## Additional Resources

- **Raw Videos**: The raw video files (144 GB) are available separately from the download page. These are required for frame extraction.
- **License**: Please review the license terms on the download page. The dataset is for **non-commercial, research purposes only**.
- **Citation**: If you use this dataset, please cite the original YouCook2 paper (see the download page for citation information).

## Troubleshooting

- If extraction fails, ensure the `.tar.gz` files are in the `data/YouCookII/` folder
- Make sure you have sufficient disk space (the extracted files are smaller than the archives)
- If you encounter permission errors, ensure you have write permissions in the folder

