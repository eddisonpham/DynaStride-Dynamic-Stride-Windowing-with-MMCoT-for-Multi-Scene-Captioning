# Results Directory

This directory contains evaluation results for both baseline and dynastride experiments.

## Structure

```
results/
├── baseline_results/
│   └── evaluation_results/
│       ├── [model_name]/
│       │   ├── [frame_sparsity]/          # Contains individual JSON result files
│       │   └── [frame_sparsity]_results.txt   # Aggregated results (avg ± std)
│       └── ...
├── dynastride_results/
│   └── evaluation_results/
│       ├── [model_name]/
│       │   ├── [frame_sparsity]/          # Contains individual JSON result files
│       │   └── [frame_sparsity]_results.txt   # Aggregated results (avg ± std)
│       └── ...
└── generate_results_txt.py                # Unified script to generate results.txt files
```

## Generating Results Summary Files

The `generate_results_txt.py` script automatically processes all evaluation results and generates aggregated summary files (`[frame_sparsity]_results.txt`) for each model and frame sparsity combination.

### Usage

Run from the `results/` directory:

```bash
python generate_results_txt.py
```

### What it does

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

For example:
```
BLEU4: avg = 4.1838, std = 0.0701
METEOR: avg = 24.3140, std = 0.1008
CIDEr: avg = 0.5559, std = 0.0040
```

### Notes

- The script automatically skips non-directory items and hidden files
- NaN values in metrics are automatically filtered out
- The script processes all available frame sparsities and models without requiring manual configuration

