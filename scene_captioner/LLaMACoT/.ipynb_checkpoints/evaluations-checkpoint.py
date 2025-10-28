import os

import subprocess
import sys

def install_dependencies():
    commands = [
        # Core ML & NLP libraries
        "pip install torch==2.2.2 transformers==4.41.2 sentence-transformers==2.7.0 bert-score==0.3.13 fastdtw==0.3.4",
        # Upgrade numpy
        "pip install --upgrade numpy==1.26.4",
        # Additional metrics and utilities
        "pip install sacrebleu==2.4.2 nltk==3.9.1 bert-score==0.3.13 pycocoevalcap==1.2 pandas",
        # NLTK data
        "python -m nltk.downloader wordnet omw-1.4 punkt"
    ]

    for cmd in commands:
        print(f"Running: {cmd}")
        subprocess.check_call(cmd, shell=True)
install_dependencies()
import sys
import json
import re
import string
import unicodedata
import glob
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Union

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider

from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util as sbert_util
from transformers import BertTokenizer, BertForNextSentencePrediction
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastdtw import fastdtw


# ----------------- Utility Functions -----------------

def normalize(s: str) -> str:
    _PUNCT_TABLE = str.maketrans("", "", string.punctuation)
    s = unicodedata.normalize("NFKC", str(s)).lower().strip().translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s)
    return s


def load_pairs_from_json(path: str) -> Tuple[
    List[str], List[str], List[str], List[Tuple[str, str]], Dict[str, List[Tuple[str, str, str]]]
]:
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
    return refs_raw, hyps_raw, ids, idx_pairs, groups


# ----------------- N-gram Metrics -----------------

def compute_ngram_metrics(refs_raw: List[str], hyps_raw: List[str]) -> Dict[str, Any]:
    refs_norm = [normalize(s) for s in refs_raw]
    hyps_norm = [normalize(s) for s in hyps_raw]

    # BLEU-4
    bleu_refs_tok = [[r.split()] for r in refs_norm]
    bleu_hyps_tok = [h.split() for h in hyps_norm]
    smooth = SmoothingFunction().method4
    bleu4 = corpus_bleu(bleu_refs_tok, bleu_hyps_tok, smoothing_function=smooth) * 100

    # METEOR
    meteor_scores = [meteor_score([r.split()], h.split()) for r, h in zip(refs_norm, hyps_norm)]
    meteor = 100 * (sum(meteor_scores) / len(meteor_scores))

    # CIDEr
    gts = {i: [refs_norm[i]] for i in range(len(refs_norm))}
    res = {i: [hyps_norm[i]] for i in range(len(hyps_norm))}
    cider_mean, _ = Cider().compute_score(gts, res)

    return {
        "BLEU4": bleu4,
        "METEOR": meteor,
        "CIDEr": float(cider_mean),
    }


# ----------------- Semantic Metrics -----------------

def compute_bertscore(hyps: List[str], refs: List[str], device: str) -> Tuple[float, float, float]:
    P, R, F1 = bert_score(hyps, refs, lang="en", rescale_with_baseline=True, verbose=False, device=device)
    return float(P.mean()), float(R.mean()), float(F1.mean())


def compute_sbert_sim(model: SentenceTransformer, hyps: List[str], refs: List[str]) -> float:
    e_h = model.encode(hyps, convert_to_tensor=True, normalize_embeddings=True)
    e_r = model.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
    sims = sbert_util.cos_sim(e_h, e_r).diagonal()
    return float(sims.mean().item())


def compute_nsp(tok, mdl, groups: Dict[str, List[Tuple[str, str, str]]], device: str) -> Tuple[float, float, float]:
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
    mdl.eval().to(device)
    total, contrad = 0, 0
    for _, seq in groups.items():
        preds = [p for _, _, p in seq if p]
        for a, b in zip(preds[:-1], preds[1:]):
            enc = tok(a, b, truncation=True, max_length=256, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = mdl(**enc).logits[0]
            label = int(torch.argmax(logits).item())  # 0=entail, 1=neutral, 2=contradiction
            total += 1
            if label == 2:
                contrad += 1
    return float(contrad / total) if total else float("nan")


# ----------------- Evaluation -----------------

def evaluate_file(args):
    json_path, device, save_path = args
    refs_raw, hyps_raw, _, _, groups = load_pairs_from_json(json_path)

    # N-gram
    ngram_metrics = compute_ngram_metrics(refs_raw, hyps_raw)

    # Semantic
    p_mean, r_mean, f_mean = compute_bertscore(hyps_raw, refs_raw, device)

    sbert_model = SentenceTransformer("all-mpnet-base-v2", device=device)
    sbert_sim = compute_sbert_sim(sbert_model, hyps_raw, refs_raw)
    dtw_align = compute_dtw_alignment(sbert_model, groups)

    nsp_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    nsp_mdl = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    nsp_true, nsp_shuf, nsp_delta = compute_nsp(nsp_tok, nsp_mdl, groups, device)

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

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[GPU {device}] Saved results to {save_path}")
    return save_path

def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    json_files = [os.path.join(d, "validation_results.json")
                  for d in sorted(glob.glob("outputs*"))
                  if os.path.exists(os.path.join(d, "validation_results.json"))]

    if not json_files:
        print("No validation_results.json found in outputs* folders.")
        sys.exit(1)

    # Build job args, skip if result already exists
    jobs = []
    for i, path in enumerate(json_files):
        save_path = os.path.join(results_dir, f"{os.path.basename(os.path.dirname(path))}.json")
        if os.path.exists(save_path):
            print(f"Skipping {path}, results already exist at {save_path}")
            continue
        gpu_id = i % 6  # round-robin assignment
        device = f"cuda:{gpu_id}"
        jobs.append((path, device, save_path))

    if not jobs:
        print("All results already exist, nothing to process.")
        return

    # Run in parallel with up to 6 processes
    with mp.Pool(processes=6) as pool:
        pool.map(evaluate_file, jobs)



if __name__ == "__main__":
    main()

