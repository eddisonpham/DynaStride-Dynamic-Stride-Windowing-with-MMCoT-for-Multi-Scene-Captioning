import os
import subprocess
import sys
import argparse

def install_dependencies():
    commands = [
        "pip install torch==2.2.2 transformers==4.41.2 sentence-transformers==2.7.0 bert-score==0.3.13 fastdtw==0.3.4",
        "pip install --upgrade numpy==1.26.4",
        "pip install sacrebleu==2.4.2 nltk==3.9.1 bert-score==0.3.13 pycocoevalcap==1.2 pandas",
        "python -m nltk.downloader wordnet omw-1.4 punkt"
    ]
    for cmd in commands:
        print(f"Running: {cmd}")
        subprocess.check_call(cmd, shell=True)

# install_dependencies()
import json
import re
import string
import unicodedata
import multiprocessing as mp
from typing import List, Tuple, Dict, Any

ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))

import torch
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider

from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util as sbert_util
from transformers import BertTokenizer, BertForNextSentencePrediction
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastdtw import fastdtw

def normalize(s: str) -> str:
    _PUNCT_TABLE = str.maketrans("", "", string.punctuation)
    s = unicodedata.normalize("NFKC", str(s)).lower().strip().translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s)
    return s

def load_pairs_from_json(path: str) -> Tuple[
    List[str], List[str], List[str], List[Tuple[str, str]], Dict[str, List[Tuple[str, str, str]]]
]:
    print(f"Loading data from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    refs_raw, hyps_raw, ids, idx_pairs = [], [], [], []
    groups: Dict[str, List[Tuple[str, str, str]]] = {}

    for vid, scenes in data.items():
        seq = []
        for sid, obj in scenes.items():
            gt = (obj.get("ground_truth") or "").strip()
            pr = (obj.get("predicted") or "").strip()
            if gt == "":
                continue
            refs_raw.append(gt)
            hyps_raw.append(pr)
            ids.append(f"{vid}|{sid}")
            idx_pairs.append((vid, sid))
            seq.append((sid, gt, pr))
        if seq:
            def _k(x: str):
                try:
                    return (0, int(x))
                except:
                    return (1, x)
            seq.sort(key=lambda t: _k(t[0]))
            groups[vid] = seq

    if not refs_raw:
        raise ValueError("No ground-truth captions found in JSON.")
    print(f"Loaded {len(refs_raw)} caption pairs from {path}.")
    return refs_raw, hyps_raw, ids, idx_pairs, groups

def compute_ngram_metrics(refs_raw: List[str], hyps_raw: List[str]) -> Dict[str, Any]:
    print("Computing n-gram metrics (BLEU4, METEOR, CIDEr)...")
    refs_norm = [normalize(s) for s in refs_raw]
    hyps_norm = [normalize(s) for s in hyps_raw]

    bleu_refs_tok = [[r.split()] for r in refs_norm]
    bleu_hyps_tok = [h.split() for h in hyps_norm]
    smooth = SmoothingFunction().method4
    bleu4 = corpus_bleu(bleu_refs_tok, bleu_hyps_tok, smoothing_function=smooth) * 100

    meteor_scores = [meteor_score([r.split()], h.split()) for r, h in zip(refs_norm, hyps_norm)]
    meteor = 100 * (sum(meteor_scores) / len(meteor_scores))

    gts = {i: [refs_norm[i]] for i in range(len(refs_norm))}
    res = {i: [hyps_norm[i]] for i in range(len(hyps_norm))}
    cider_mean, _ = Cider().compute_score(gts, res)

    return {
        "BLEU4": bleu4,
        "METEOR": meteor,
        "CIDEr": float(cider_mean),
    }

def compute_bertscore(hyps: List[str], refs: List[str], device: str) -> Tuple[float, float, float]:
    print("Computing BERTScore metrics...")
    P, R, F1 = bert_score(hyps, refs, lang="en", rescale_with_baseline=True, verbose=False, device=device)
    return float(P.mean()), float(R.mean()), float(F1.mean())

def compute_sbert_sim(model: SentenceTransformer, hyps: List[str], refs: List[str]) -> float:
    print("Computing SBERT similarity...")
    e_h = model.encode(hyps, convert_to_tensor=True, normalize_embeddings=True)
    e_r = model.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
    sims = sbert_util.cos_sim(e_h, e_r).diagonal()
    return float(sims.mean().item())

def compute_nsp(tok, mdl, groups: Dict[str, List[Tuple[str, str, str]]], device: str) -> Tuple[float, float, float]:
    print("Computing temporal coherence (Next Sentence Prediction, NSP)...")
    mdl.eval().to(device)
    true_scores, shuf_scores = [], []
    import random

    for _, seq in groups.items():
        preds = [p for _, _, p in seq if p]
        if len(preds) < 2:
            continue
        for a, b in zip(preds[:-1], preds[1:]):
            enc = tok(a, b, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                probs = torch.softmax(mdl(**enc).logits, dim=1)[0]
            true_scores.append(float(probs[0].item()))
        shuf = preds[:]
        random.shuffle(shuf)
        for a, b in zip(shuf[:-1], shuf[1:]):
            enc = tok(a, b, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                probs = torch.softmax(mdl(**enc).logits, dim=1)[0]
            shuf_scores.append(float(probs[0].item()))

    if not true_scores or not shuf_scores:
        return float("nan"), float("nan"), float("nan")
    true_m = sum(true_scores) / len(true_scores)
    shuf_m = sum(shuf_scores) / len(shuf_scores)
    return true_m, shuf_m, (true_m - shuf_m)

def compute_dtw_alignment(model: SentenceTransformer, groups: Dict[str, List[Tuple[str, str, str]]]) -> float:
    print("Computing temporal alignment (DTW)...")
    dists = []
    for _, seq in groups.items():
        gts = [g for _, g, _ in seq if g]
        hyps = [p for _, _, p in seq if p]
        if not gts or not hyps:
            continue
        e_g = model.encode(gts, convert_to_tensor=False, normalize_embeddings=True)
        e_h = model.encode(hyps, convert_to_tensor=False, normalize_embeddings=True)

        def cos_dist(u, v):
            return 1.0 - float(sum(a * b for a, b in zip(u, v)))

        d, _ = fastdtw(e_h, e_g, dist=cos_dist)
        dists.append(float(d))
    return float(sum(dists) / len(dists)) if dists else float("nan")

def compute_nli_contradiction_rate(tok, mdl, groups: Dict[str, List[Tuple[str, str, str]]], device: str) -> float:
    print("Computing temporal contradiction rate (NLI)...")
    mdl.eval().to(device)
    total, contrad = 0, 0
    for _, seq in groups.items():
        preds = [p for _, _, p in seq if p]
        for a, b in zip(preds[:-1], preds[1:]):
            enc = tok(a, b, truncation=True, max_length=256, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = mdl(**enc).logits[0]
            label = int(torch.argmax(logits).item())
            total += 1
            if label == 2:
                contrad += 1
    return float(contrad / total) if total else float("nan")

def evaluate_file(args):
    json_path, device, save_path = args
    print(f"-----\nEvaluating: {json_path}\nDevice: {device}\nOutput: {save_path}")
    refs_raw, hyps_raw, _, _, groups = load_pairs_from_json(json_path)

    print("Stage 1: N-gram metrics...")
    ngram_metrics = compute_ngram_metrics(refs_raw, hyps_raw)

    print("Stage 2: BERTScore...")
    p_mean, r_mean, f_mean = compute_bertscore(hyps_raw, refs_raw, device)

    print("Stage 3: SBERT and DTW computation...")
    sbert_model = SentenceTransformer("all-mpnet-base-v2", device=device)
    sbert_sim = compute_sbert_sim(sbert_model, hyps_raw, refs_raw)
    dtw_align = compute_dtw_alignment(sbert_model, groups)

    print("Stage 4: Temporal coherence (NSP)...")
    nsp_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    nsp_mdl = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    nsp_true, nsp_shuf, nsp_delta = compute_nsp(nsp_tok, nsp_mdl, groups, device)

    print("Stage 5: Temporal contradiction rate (NLI)...")
    nli_tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
    nli_mdl = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    nli_contrad = compute_nli_contradiction_rate(nli_tok, nli_mdl, groups, device)

    metrics = {
        "num_samples": len(refs_raw),
        **ngram_metrics,
        "PBERT": p_mean,
        "RBERT": r_mean,
        "FBERT": f_mean,
        "SBERTSim": sbert_sim,
        "TemporalCoherence_NSP_true": nsp_true,
        "TemporalCoherence_NSP_shuffled": nsp_shuf,
        "TemporalCoherence_NSP_delta": nsp_delta,
        "TemporalAlignment_DTW": dtw_align,
        "TemporalContradictionRate_NLI": nli_contrad,
    }

    print("Saving results...")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Device {device}] Saved results to {save_path}")
    return save_path

def parse_run_name(run_name: str, model_name: str) -> Tuple[str, str]:
    """
    Parse run_name like 'outputs_mistral_10f_seed1' to extract frame_sparsity and seed.
    Returns: (frame_sparsity, seed_number)
    """
    # Pattern: outputs_[model_name]_[frame_sparsity]f_seed[seed_number]
    # Remove 'outputs_' prefix
    if run_name.startswith('outputs_'):
        rest = run_name[8:]  # Remove 'outputs_'
    else:
        rest = run_name
    
    # Remove model_name prefix
    if rest.startswith(f"{model_name}_"):
        rest = rest[len(model_name) + 1:]  # Remove 'model_name_'
    
    # Match pattern: [frame_sparsity]f_seed[seed_number]
    match = re.match(r'^(\d+)f_seed(\d+)$', rest)
    if match:
        frame_sparsity = match.group(1)
        seed_number = match.group(2)
        return frame_sparsity, seed_number
    
    # Fallback: try to extract numbers if pattern doesn't match exactly
    print(f"Warning: Could not parse run_name '{run_name}' with expected pattern. Trying fallback...")
    # Look for pattern like "10f_seed1" or similar
    match = re.search(r'(\d+)f.*?seed(\d+)', rest)
    if match:
        return match.group(1), match.group(2)
    
    raise ValueError(f"Could not parse run_name '{run_name}' to extract frame_sparsity and seed")

def main():
    parser = argparse.ArgumentParser(description="Evaluate scene caption outputs with optional GPU/multiprocessing support")
    parser.add_argument(
        "--use_gpu_queue",
        default=False,
        action="store_true", 
        help="Distribute evaluation over multiple GPUs in parallel if available"
    )
    parser.add_argument(
        "--num_gpus",
        type=int, default=6,
        help="Number of GPUs to use if --use_gpu_queue is enabled"
    )
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    results_dir = os.path.join(root, "results", "dynastride_results")
    os.makedirs(results_dir, exist_ok=True)

    eval_outputs_dir = os.path.join(results_dir, "outputs")
    json_files = []
    print(f"Searching for validation_results.json in: {eval_outputs_dir}/<model>/<run>/")
    if os.path.isdir(eval_outputs_dir):
        for model_name in sorted(os.listdir(eval_outputs_dir)):
            model_dir = os.path.join(eval_outputs_dir, model_name)
            if not os.path.isdir(model_dir):
                continue
            for run_name in sorted(os.listdir(model_dir)):
                run_dir = os.path.join(model_dir, run_name)
                json_path = os.path.join(run_dir, "validation_results.json")
                if os.path.isdir(run_dir) and os.path.exists(json_path):
                    json_files.append((model_name, run_name, json_path))
    if not json_files:
        print("No validation_results.json found in dynastride_results/outputs/*/* folders.")
        sys.exit(1)

    print(f"Found {len(json_files)} models/runs ready for evaluation.")
    jobs = []
    eval_results_dir = os.path.join(results_dir, "evaluation_results")
    os.makedirs(eval_results_dir, exist_ok=True)
    for idx, (model_name, run_name, json_path) in enumerate(json_files):
        print(f"Processing {model_name}/{run_name}...")
        try:
            # Parse run_name to extract frame_sparsity and seed
            frame_sparsity, seed_number = parse_run_name(run_name, model_name)
            print(f"Parsed frame_sparsity: {frame_sparsity}, seed_number: {seed_number}")
            
            # Create folder structure: evaluation_results/[model_name]/[frame_sparsity]/
            save_model_dir = os.path.join(eval_results_dir, model_name)
            save_frame_dir = os.path.join(save_model_dir, frame_sparsity)
            os.makedirs(save_frame_dir, exist_ok=True)
            
            # Output filename: outputs_[model_name]_[frame_sparsity]_seed[seed_number].json
            output_filename = f"outputs_{model_name}_{frame_sparsity}_seed{seed_number}.json"
            save_path = os.path.join(save_frame_dir, output_filename)
            
            if os.path.exists(save_path):
                print(f"Skipping {json_path}, results already exist at {save_path}")
                continue
                
            if args.use_gpu_queue and torch.cuda.is_available():
                gpu_id = idx % args.num_gpus
                device = f"cuda:{gpu_id}"
            else:
                device = "cpu"
            jobs.append((json_path, device, save_path))
        except ValueError as e:
            print(f"Error parsing run_name '{run_name}': {e}. Skipping...")
            continue

    if not jobs:
        print("All results already exist, nothing to process.")
        return

    print(f"Starting evaluation for {len(jobs)} jobs...")
    use_gpu_queue = args.use_gpu_queue and torch.cuda.is_available()
    if use_gpu_queue:
        print(f"Multiprocessing with {args.num_gpus} GPUs (gpu queue)...")
        with mp.Pool(processes=args.num_gpus) as pool:
            pool.map(evaluate_file, jobs)
    else:
        for job in jobs:
            evaluate_file(job)

if __name__ == "__main__":
    main()
