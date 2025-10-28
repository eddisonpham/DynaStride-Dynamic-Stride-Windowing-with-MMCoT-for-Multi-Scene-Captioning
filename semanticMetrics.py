import os
import subprocess
import sys

def install_dependencies():
    commands = [
        # DO NOT install/uninstall torch here; use the container's GPU build or your venv's GPU build.
        "pip install transformers==4.41.2 sentence-transformers==2.7.0 bert-score==0.3.13 fastdtw==0.3.4",
        "pip install --upgrade numpy==1.26.4"
    ]
    for cmd in commands:
        print(f"Running: {cmd}")
        subprocess.check_call(cmd, shell=True)

install_dependencies()

import json
import re
import string
import unicodedata
import glob
import multiprocessing as mp
from typing import List, Tuple, Dict

import torch
import torch.multiprocessing as tmp
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util as sbert_util
from transformers import BertTokenizer, BertForNextSentencePrediction
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastdtw import fastdtw

# ---- CUDA-safe multiprocessing ----
tmp.set_start_method("spawn", force=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---- Detect visible GPUs (honors CUDA_VISIBLE_DEVICES) ----
num_gpus = torch.cuda.device_count()  # e.g., 5 if you export CUDA_VISIBLE_DEVICES=0,2,3,4,5
assert num_gpus >= 1, "No visible GPUs!"
print(f"[Sanity] torch={torch.__version__}, compiled CUDA={torch.version.cuda}")
print(f"[Sanity] Visible GPUs (NVML): {num_gpus}")

# ----------------- Utility Functions -----------------

def normalize(s: str) -> str:
    _PUNCT_TABLE = str.maketrans("", "", string.punctuation)
    s = unicodedata.normalize("NFKC", str(s)).lower().strip().translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s)
    return s

def load_pairs_from_json(path: str) -> Tuple[
    List[str], List[str], List[str], List[Tuple[str, str]], Dict[str, List[Tuple[str, str, str]]]
]:
    """
    Expects:
    {
      "video_id": {
         "scene_id": {"ground_truth": "...", "predicted": "..."},
         ...
      }
    }
    """
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
        raise ValueError(f"No ground-truth captions found in JSON: {path}")
    return refs_raw, hyps_raw, ids, idx_pairs, groups

# ----------------- Metrics -----------------

def compute_bertscore(hyps: List[str], refs: List[str], device: str):
    P, R, F1 = bert_score(hyps, refs, lang="en", rescale_with_baseline=True, verbose=False, device=device)
    return float(P.mean()), float(R.mean()), float(F1.mean())

def compute_sbert_sim(model: SentenceTransformer, hyps: List[str], refs: List[str]) -> float:
    e_h = model.encode(hyps, convert_to_tensor=True, normalize_embeddings=True)
    e_r = model.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
    sims = sbert_util.cos_sim(e_h, e_r).diagonal()
    return float(sims.mean().item())

def compute_dtw_alignment(model: SentenceTransformer, groups: Dict[str, List[Tuple[str, str, str]]]) -> float:
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

def compute_nsp(tok, mdl, groups: Dict[str, List[Tuple[str, str, str]]], device: str):
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

def compute_nli_contradiction_rate(tok, mdl, groups: Dict[str, List[Tuple[str, str, str]]], device: str) -> float:
    mdl.eval().to(device)
    total, contrad = 0, 0
    for _, seq in groups.items():
        preds = [p for _, _, p in seq if p]
        for a, b in zip(preds[:-1], preds[1:]):
            enc = tok(a, b, truncation=True, max_length=256, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = mdl(**enc).logits[0]
            label = int(torch.argmax(logits).item())  # 0 ent, 1 neutral, 2 contradiction
            total += 1
            if label == 2:
                contrad += 1
    return float(contrad / total) if total else float("nan")

# ----------------- Worker -----------------

def evaluate_file(args):
    json_path, device, save_path = args

    # Bind GPU BEFORE loading any models
    try:
        if device.startswith("cuda"):
            idx = int(device.split(":")[1])
            torch.cuda.set_device(idx)
            # optional debug print
            print(f"[worker] using {device} -> {torch.cuda.get_device_name(idx)}")
    except Exception as e:
        print(f"[{device}] set_device failed: {e} — falling back to CPU")
        device = "cpu"

    refs_raw, hyps_raw, _, _, groups = load_pairs_from_json(json_path)

    # BERTScore (GPU → CPU fallback if needed)
    try:
        p_mean, r_mean, f_mean = compute_bertscore(hyps_raw, refs_raw, device)
    except Exception as e:
        print(f"[{device}] BERTScore failed: {e} — retrying on CPU")
        p_mean, r_mean, f_mean = compute_bertscore(hyps_raw, refs_raw, "cpu")

    # SBERT + DTW
    try:
        sbert_model = SentenceTransformer("all-mpnet-base-v2", device=device if device != "cpu" else None)
    except Exception as e:
        print(f"[{device}] SBERT init failed: {e} — retrying on CPU")
        sbert_model = SentenceTransformer("all-mpnet-base-v2", device=None)
        device = "cpu"
    sbert_sim = compute_sbert_sim(sbert_model, hyps_raw, refs_raw)
    dtw_align = compute_dtw_alignment(sbert_model, groups)

    # NSP
    try:
        nsp_tok = BertTokenizer.from_pretrained("bert-base-uncased")
        nsp_mdl = BertForNextSentencePrediction.from_pretrained("bert-base-uncased").to(device)
        nsp_true, nsp_shuf, nsp_delta = compute_nsp(nsp_tok, nsp_mdl, groups, device)
    except Exception as e:
        print(f"[{device}] NSP failed: {e} — setting NSP metrics to NaN")
        nsp_true = nsp_shuf = nsp_delta = float("nan")

    # NLI
    try:
        nli_tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
        nli_mdl = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)
        nli_contrad = compute_nli_contradiction_rate(nli_tok, nli_mdl, groups, device)
    except Exception as e:
        print(f"[{device}] NLI failed: {e} — setting contradiction rate to NaN")
        nli_contrad = float("nan")

    metrics = {
        "num_samples": len(refs_raw),
        "PBERT": p_mean, "RBERT": r_mean, "FBERT": f_mean,
        "SBERTSim": sbert_sim,
        "TemporalCoherence_NSP_true": nsp_true,
        "TemporalCoherence_NSP_shuffled": nsp_shuf,
        "TemporalCoherence_NSP_delta": nsp_delta,
        "TemporalAlignment_DTW": dtw_align,
        "TemporalContradictionRate_NLI": nli_contrad,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{device}] Saved results to {save_path}")
    return save_path

# ----------------- Main -----------------

def main():
    results_dir = "SemanticResults"
    os.makedirs(results_dir, exist_ok=True)
    json_files = sorted(glob.glob(os.path.join("outputs/llama3", "*.json")))
    if not json_files:
        print("No JSON files found in ./llama3.")
        sys.exit(1)

    jobs = []
    for i, path in enumerate(json_files):
        base = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(results_dir, f"{base}.json")
        if os.path.exists(save_path):
            print(f"Skipping {path}, results already exist at {save_path}")
            continue
        gpu_id = i % num_gpus                  # <-- use detected GPU count
        device = f"cuda:{gpu_id}"
        jobs.append((path, device, save_path))

    if not jobs:
        print("All results already exist, nothing to process.")
        return

    print(f"Dispatching {len(jobs)} job(s) across {num_gpus} visible GPU(s) with spawn")
    # Use as many workers as visible GPUs; maxtasksperchild helps free memory per task
    with mp.Pool(processes=num_gpus, maxtasksperchild=1) as pool:
        pool.map(evaluate_file, jobs)

if __name__ == "__main__":
    main()
