# Results Directory

This directory contains evaluation results for both baseline and DynaStride experiments.

## Directory Structure

```
results/
├── baseline_results/              # Baseline model results (GPT-4o, VideoLLaMA3)
│   ├── outputs/                   # Raw caption outputs (JSON files)
│   │   ├── gpt4o/                 # GPT-4o captions
│   │   │   ├── gpt40Captions.json
│   │   │   ├── gpt40Captions10.json
│   │   │   └── ...
│   │   └── llama3/                 # VideoLLaMA3 captions
│   │       ├── llama3Captions10.json
│   │       └── ...
│   │
│   └── evaluation_results/         # Evaluated metrics (JSON + aggregated TXT)
│       ├── gpt/                   # GPT-4o evaluation results
│       │   ├── 5/                  # Frame sparsity = 5
│       │   │   ├── gpt40Captions.json
│       │   │   ├── gptCaptionsRound2.json
│       │   │   └── gptCaptionsRound3.json
│       │   ├── 5_results.txt       # Aggregated statistics
│       │   ├── 10/
│       │   ├── 10_results.txt
│       │   └── ...
│       └── llama3/                 # VideoLLaMA3 evaluation results
│           └── ...
│
├── dynastride_results/             # DynaStride pipeline results
│   ├── outputs/                    # Raw caption outputs
│   │   ├── qwen/                   # Qwen model outputs
│   │   │   ├── outputs_qwen_5f_seed1/
│   │   │   │   └── validation_results.json
│   │   │   ├── outputs_qwen_10f_seed1/
│   │   │   └── ...
│   │   ├── mistral/                # Mistral model outputs
│   │   └── phi/                    # Phi model outputs
│   │
│   └── evaluation_results/         # Evaluated metrics
│       ├── qwen/                   # Qwen evaluation results
│       │   ├── 5/
│       │   │   ├── outputs_qwen_5f_seed1.json
│       │   │   ├── outputs_qwen_5f_seed2.json
│       │   │   └── outputs_qwen_5f_seed3.json
│       │   ├── 5_results.txt       # Aggregated statistics
│       │   ├── 10/
│       │   ├── 10_results.txt
│       │   └── ...
│       ├── mistral/
│       └── phi/
│
└── generate_results_txt.py         # Script to generate aggregated results
```

## File Types

### Raw Output Files (`.json`)

Located in `outputs/` directories, these contain the raw caption predictions:

```json
{
  "video_id": {
    "scene_index": {
      "ground_truth": "reference caption",
      "predicted": "generated caption"
    }
  }
}
```

### Evaluation Result Files (`.json`)

Located in `evaluation_results/[model]/[frame_sparsity]/`, these contain computed metrics:

```json
{
  "num_samples": 1234,
  "BLEU4": 4.1838,
  "METEOR": 24.3140,
  "CIDEr": 0.5559,
  "PBERT": 0.8234,
  "RBERT": 0.7891,
  "FBERT": 0.8056,
  "SBERTSim": 0.7123,
  "TemporalCoherence_NSP_true": 0.6543,
  "TemporalCoherence_NSP_shuffled": 0.5234,
  "TemporalCoherence_NSP_delta": 0.1309,
  "TemporalAlignment_DTW": 1.2345,
  "TemporalContradictionRate_NLI": 0.1234
}
```

### Aggregated Results Files (`[frame_sparsity]_results.txt`)

Located in `evaluation_results/[model]/`, these contain average and standard deviation across multiple runs:

```
BLEU4: avg = 4.1838, std = 0.0701
METEOR: avg = 24.3140, std = 0.1008
CIDEr: avg = 0.5559, std = 0.0040
PBERT: avg = 0.8234, std = 0.0123
RBERT: avg = 0.7891, std = 0.0098
FBERT: avg = 0.8056, std = 0.0105
SBERTSim: avg = 0.7123, std = 0.0089
TemporalCoherence_NSP_true: avg = 0.6543, std = 0.0234
TemporalCoherence_NSP_shuffled: avg = 0.5234, std = 0.0198
TemporalCoherence_NSP_delta: avg = 0.1309, std = 0.0156
TemporalAlignment_DTW: avg = 1.2345, std = 0.1234
TemporalContradictionRate_NLI: avg = 0.1234, std = 0.0098
```

## Metrics Explained

### N-gram Metrics

- **BLEU4**: BLEU score with 4-gram precision (higher is better, range: 0-100)
- **METEOR**: METEOR score considering synonyms and paraphrases (higher is better, range: 0-100)
- **CIDEr**: Consensus-based Image Description Evaluation (higher is better, typically 0-1)

### Semantic Similarity Metrics

- **PBERT**: BERTScore Precision (higher is better, range: 0-1)
- **RBERT**: BERTScore Recall (higher is better, range: 0-1)
- **FBERT**: BERTScore F1 (higher is better, range: 0-1)
- **SBERTSim**: SBERT (Sentence-BERT) cosine similarity (higher is better, range: 0-1)

### Temporal Coherence Metrics

- **TemporalCoherence_NSP_true**: Next Sentence Prediction score for true sequence order (higher is better, range: 0-1)
- **TemporalCoherence_NSP_shuffled**: NSP score for shuffled sequence order (lower is better, range: 0-1)
- **TemporalCoherence_NSP_delta**: Difference between true and shuffled NSP scores (higher is better, indicates better temporal coherence)
- **TemporalAlignment_DTW**: Dynamic Time Warping distance between predicted and ground truth sequences (lower is better)
- **TemporalContradictionRate_NLI**: Rate of contradictions between consecutive captions (lower is better, range: 0-1)

## Generating Aggregated Results

The `generate_results_txt.py` script automatically processes all evaluation results and generates aggregated summary files.

### Usage

Run from the `results/` directory:

```bash
cd results
python generate_results_txt.py
```

### What It Does

1. **Processes both result types**: Automatically processes both `baseline_results/evaluation_results/` and `dynastride_results/evaluation_results/`

2. **Finds all models**: Discovers all model directories (e.g., `gpt`, `llama3`, `mistral`, `phi`, `qwen`)

3. **Processes frame sparsity folders**: For each model, finds all frame sparsity subdirectories (e.g., `5`, `10`, `20`, `40`)

4. **Aggregates metrics**: Collects all JSON result files in each frame sparsity folder and computes:
   - Average (mean) for each metric
   - Standard deviation for each metric

5. **Generates summary files**: Creates `[frame_sparsity]_results.txt` files in each model directory containing aggregated statistics

### Output Format

The generated `[frame_sparsity]_results.txt` files contain lines in the format:
```
metric_name: avg = X.XXXX, std = X.XXXX
```

### Notes

- The script automatically skips non-directory items and hidden files
- NaN values in metrics are automatically filtered out
- The script processes all available frame sparsities and models without requiring manual configuration
- Existing result files are not overwritten; the script will skip processing if results already exist

## Interpreting Results

### Comparing Models

To compare different models, look at the aggregated results files:

```bash
# Compare GPT-4o vs DynaStride (Qwen) at 10 frames
cat baseline_results/evaluation_results/gpt/10_results.txt
cat dynastride_results/evaluation_results/qwen/10_results.txt
```

### Frame Sparsity Analysis

To analyze the effect of frame sparsity, compare results across different sparsity levels:

```bash
# Compare Qwen at different frame sparsities
cat dynastride_results/evaluation_results/qwen/5_results.txt
cat dynastride_results/evaluation_results/qwen/10_results.txt
cat dynastride_results/evaluation_results/qwen/20_results.txt
cat dynastride_results/evaluation_results/qwen/40_results.txt
```

### Reproducibility

Multiple runs (seeds) are stored separately in the `[frame_sparsity]/` directories. The aggregated results show the mean and standard deviation across these runs, providing confidence intervals for the metrics.

## File Naming Conventions

### Baseline Outputs

- `gpt40Captions.json`: GPT-4o with default frame sparsity
- `gpt40Captions[sparsity].json`: GPT-4o with specific frame sparsity
- `gpt40Captions[sparsity]Round[round].json`: Multiple rounds for reproducibility
- `llama3Captions[sparsity].json`: VideoLLaMA3 outputs

### DynaStride Outputs

- `outputs_[model]_[sparsity]f_seed[seed]/`: Directory containing outputs for a specific model, frame sparsity, and seed
- `outputs_[model]_[sparsity]f_seed[seed].json`: Evaluation results file

### Evaluation Results

- `outputs_[model]_[sparsity]f_seed[seed].json`: Individual evaluation result
- `[sparsity]_results.txt`: Aggregated results across all seeds

## Troubleshooting

### Missing Results

If results are missing:
1. Check that the evaluation script completed successfully
2. Verify that the output JSON files exist in the `outputs/` directories
3. Ensure the evaluation script was run with the correct paths

### Inconsistent Metrics

If metrics seem inconsistent:
1. Check that all JSON files in a frame sparsity folder are from the same model and configuration
2. Verify that NaN values are being handled correctly (they are automatically filtered)
3. Ensure sufficient number of samples for statistical significance

### Regenerating Results

To regenerate aggregated results:
1. Delete the `[frame_sparsity]_results.txt` files you want to regenerate
2. Run `python generate_results_txt.py` again
