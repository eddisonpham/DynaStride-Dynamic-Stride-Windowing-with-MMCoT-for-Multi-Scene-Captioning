# DynaStride: Dynamic Stride Windowing with MMCoT for Multi-Scene Captioning

**Paper**: [arXiv:2510.23907](https://arxiv.org/abs/2510.23907)

Accepted for oral presentation:
@ [NeurIPS 7HVU 2025](https://holistic-video-understanding.github.io/workshops/neurips2025.html)
@ [AAAI AI4ED 2026](https://ai4ed.cc/workshops/aaai2026)

## Abstract Summary

Scene-level captioning in instructional videos enhances learning by requiring understanding of both visual cues and temporal structure. However, captions that fail to capture this structure may lack coherence and quality, undermining the video's educational intent. 

DynaStride addresses this gap by generating coherent, scene-level captions without requiring manual scene segmentation. The pipeline uses the YouCookII dataset's scene annotations to perform adaptive frame sampling and multimodal windowing, capturing key transitions within each scene. It employs a multimodal chain-of-thought (MMCoT) process to produce multiple action-object pairs, which are refined and fused using a dynamic stride window selection algorithm that adaptively balances temporal context and redundancy. The final scene-level caption integrates visual semantics and temporal reasoning in a single instructional caption.

Empirical evaluations against strong baselines (VLLaMA3 and GPT-4o) demonstrate consistent gains on both N-gram-based metrics (BLEU, METEOR) and semantic similarity measures (BERTScore, CLIPScore). Qualitative analyses show that DynaStride produces captions that are more temporally coherent and informative.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for model inference)
- Sufficient disk space for YouCookII dataset (~144 GB for raw videos)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd DynaStride-Dynamic-Stride-Windowing-with-MMCoT-for-Multi-Scene-Captioning
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (required for evaluation metrics):
```bash
python -m nltk.downloader wordnet omw-1.4 punkt
```

4. **Set up environment variables**:
   - Create `.env` files in the following locations:
     - `src/experiments/.env` - For baseline experiments (OpenAI API key)
     - `src/scene_captioner/.env` - For DynaStride pipeline (HuggingFace token)
   
   Example `.env` files:
   ```bash
   # src/experiments/.env
   OPEN_AI_API_KEY=your_openai_api_key_here
   
   # src/scene_captioner/.env
   HUGGINGFACE_KEY=your_huggingface_token_here
   ```

5. **Download and prepare YouCookII dataset**:
   - Follow instructions in `data/YouCookII/README.md`
   - Extract annotations and splits
   - Download raw videos (optional, if you have pre-extracted frames)

## Repository Structure

```
.
├── data/                          # Dataset and preprocessing
│   ├── preprocessing/             # Data preprocessing utilities
│   │   ├── preprocessor.py        # YouCookII annotation preprocessing
│   │   └── segment_loader.py      # Segment loading utilities
│   └── YouCookII/                 # YouCookII dataset
│       ├── raw_videos/            # Raw video files (if downloaded)
│       ├── splits/                # Train/val/test splits
│       ├── scripts/               # Dataset download scripts
│       └── youcookii_annotations_trainval.json
│
├── src/                           # Source code
│   ├── baselines/                 # Baseline model implementations
│   │   ├── gpt_captioner.py       # GPT-4o captioning
│   │   └── videollama3_captioner.py  # VideoLLaMA3 captioning
│   │
│   ├── scene_captioner/           # DynaStride pipeline
│   │   ├── mcot_module.py         # Multimodal chain-of-thought module
│   │   ├── aggregator.py          # Caption aggregation (CookingAggregator)
│   │   ├── evaluate_dynastride.py # Evaluation script for DynaStride outputs
│   │   └── run_pipeline.ipynb     # Main DynaStride pipeline notebook
│   │
│   ├── experiments/               # Baseline experiment runner
│   │   └── run_experiment.py      # Unified experiment runner
│   │
│   ├── evaluation_metrics/        # Evaluation metrics
│   │   ├── n-gramMetrics.py       # BLEU, METEOR, CIDEr
│   │   └── semantic_metrics.py    # BERTScore, SBERT, temporal metrics
│   │
│   └── utils/                     # Utility functions
│       ├── frame_sampling.py      # Frame sampling strategies
│       ├── image_processing.py    # Image preprocessing
│       └── loader_utils.py        # Data loading utilities
│
├── results/                       # Experimental results
│   ├── baseline_results/          # Baseline model results
│   │   ├── outputs/               # Raw caption outputs
│   │   └── evaluation_results/   # Evaluated metrics
│   ├── dynastride_results/        # DynaStride results
│   │   ├── outputs/               # Raw caption outputs
│   │   └── evaluation_results/   # Evaluated metrics
│   └── generate_results_txt.py    # Results aggregation script
│
├── plots/                         # Visualization notebooks and graphs
│   ├── plots.ipynb                # Results visualization
│   └── graphs/                    # Generated metric plots
│
└── constants/                     # Configuration files
    ├── sampled_videos.txt         # List of videos to process
    └── plot_names.txt             # Plot naming configuration
```

## Pipeline Overview

The DynaStride pipeline consists of several stages:

1. **Data Preprocessing**: Extract and prepare YouCookII annotations
2. **Frame Sampling**: Adaptive frame sampling from video scenes
3. **Multimodal Captioning**: Generate captions using MMCoT with dynamic stride windowing
4. **Caption Aggregation**: Fuse multiple captions into scene-level summaries
5. **Evaluation**: Compute metrics (N-gram and semantic)

## Running the Pipeline

### Step 1: Data Preprocessing

Preprocess YouCookII annotations to extract ground truth references:

```bash
cd data/preprocessing
python preprocessor.py
```

This generates:
- `data/YouCookII/saved_references/youcook2_validation_refs.pkl`
- `data/YouCookII/saved_references/youcook2_validation_preds.pkl`

### Step 2: Run Baseline Experiments (Optional)

Run baseline models (GPT-4o or VideoLLaMA3) for comparison:

```bash
cd src/experiments

# Run GPT-4o with 10 frames per scene
python run_experiment.py --model gpt --max-frames 10 --round 1

# Run VideoLLaMA3 with 10 frames per scene
python run_experiment.py --model videollama3 --max-frames 10 --round 1
```

**Parameters**:
- `--model`: `gpt` or `videollama3`
- `--max-frames`: Frame sparsity (5, 10, 20, or 40)
- `--round`: Round/seed number (default: 1)

**Output**: Results saved to `results/baseline_results/outputs/[model]/`

### Step 3: Run DynaStride Pipeline

The main DynaStride pipeline is implemented in a Jupyter notebook:

```bash
# Open and run the notebook
jupyter notebook src/scene_captioner/run_pipeline.ipynb
```

**Configuration**:
- Set model ID (e.g., `Qwen/Qwen3-4B-Instruct-2507`, `mistralai/Mistral-7B-Instruct-v0.3`)
- Set frame sparsity (5, 10, 20, or 40)
- Set seed for reproducibility

**Output**: Results saved to `results/dynastride_results/outputs/[model]/outputs_[model]_[sparsity]f_seed[seed]/`

### Step 4: Evaluate Results

#### Evaluate DynaStride Results

```bash
cd src/scene_captioner

# Single GPU evaluation
python evaluate_dynastride.py

# Multi-GPU evaluation (if available)
python evaluate_dynastride.py --use_gpu_queue --num_gpus 6
```

**Output**: Metrics saved to `results/dynastride_results/evaluation_results/[model]/[frame_sparsity]/`

#### Evaluate Baseline Results

Baseline results are evaluated using the same evaluation scripts. The evaluation metrics include:

- **N-gram metrics**: BLEU-4, METEOR, CIDEr
- **Semantic metrics**: BERTScore (P/R/F1), SBERT similarity
- **Temporal metrics**: 
  - Temporal Coherence (NSP)
  - Temporal Alignment (DTW)
  - Temporal Contradiction Rate (NLI)

### Step 5: Generate Aggregated Results

Aggregate evaluation results across multiple runs:

```bash
cd results
python generate_results_txt.py
```

This generates `[frame_sparsity]_results.txt` files in each model directory containing average and standard deviation statistics.

### Step 6: Visualize Results (Optional)

```bash
jupyter notebook plots/plots.ipynb
```

## High-Level Pipeline Flow

```
1. Data Preparation
   └─> Preprocess YouCookII annotations
   └─> Extract/load video frames

2. Frame Sampling
   └─> Adaptive frame sampling per scene
   └─> Apply frame sparsity (5/10/20/40)

3. Caption Generation
   ├─> Baseline: Direct captioning (GPT-4o/VideoLLaMA3)
   └─> DynaStride:
       ├─> MMCoT: Generate action-object pairs
       ├─> Dynamic stride windowing
       └─> Caption aggregation

4. Evaluation
   └─> Compute metrics (N-gram + semantic + temporal)

5. Results Aggregation
   └─> Generate summary statistics
```

## Key Components

### DynaStride Pipeline (`src/scene_captioner/`)

- **`mcot_module.py`**: Multimodal chain-of-thought reasoning for generating action-object pairs from frame sequences
- **`aggregator.py`**: `CookingAggregator` class that fuses multiple captions into coherent scene-level summaries
- **`run_pipeline.ipynb`**: Main pipeline orchestrating frame sampling, MMCoT captioning, and aggregation

### Baseline Models (`src/baselines/`)

- **`gpt_captioner.py`**: GPT-4o API wrapper for video captioning
- **`videollama3_captioner.py`**: VideoLLaMA3 local model for captioning

### Evaluation (`src/evaluation_metrics/`)

- **`n-gramMetrics.py`**: BLEU, METEOR, CIDEr computation
- **`semantic_metrics.py`**: BERTScore, SBERT, temporal coherence metrics

## Configuration

### Frame Sparsity

Frame sparsity controls how many frames are sampled per scene:
- `5`: Every 5th frame (most dense)
- `10`: Every 10th frame (default)
- `20`: Every 20th frame
- `40`: Every 40th frame (most sparse)

### Model Selection

**Baseline Models**:
- `gpt`: GPT-4o (requires OpenAI API key)
- `videollama3`: VideoLLaMA3 (local model)

**DynaStride Models**:
- `Qwen/Qwen3-4B-Instruct-2507` (default)
- `mistralai/Mistral-7B-Instruct-v0.3`
- `microsoft/phi-2`
- Other compatible instruction-tuned models

## Results Structure

See `results/README.md` for detailed information about the results directory structure and how to interpret the evaluation outputs.

## Citation

If you use this code or find it helpful, please cite our paper:

```bibtex
@article{pham2025dynastride,
  title={DynaStride: Dynamic Stride Windowing with MMCoT for Instructional Multi-Scene Captioning},
  author={Pham, Eddison and Priyadarshini, Prisha and Maliackel, Adrian and Bandi, Kanishk and Meo, Cristian and Zhu, Kevin},
  journal={arXiv preprint arXiv:2510.23907},
  year={2025}
}
```

## License

This project is for research purposes only. Please review the YouCookII dataset license terms before use.

## Troubleshooting

### Common Issues

1. **Missing API keys**: Ensure `.env` files are properly configured
2. **CUDA out of memory**: Reduce batch size or use CPU for smaller models
3. **Missing NLTK data**: Run `python -m nltk.downloader wordnet omw-1.4 punkt`
4. **Video files not found**: Ensure YouCookII videos are downloaded and extracted

### Getting Help

For issues or questions, please open an issue on the repository or refer to the paper for detailed methodology.
