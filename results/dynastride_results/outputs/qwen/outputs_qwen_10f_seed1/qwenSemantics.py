import sys, json
from typing import List, Tuple, Dict, Any, Union

from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util as sbert_util
import torch
from transformers import BertTokenizer, BertForNextSentencePrediction
from fastdtw import fastdtw
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_pairs_from_json(path: str) -> Tuple[
    List[str], List[str], List[str], List[Tuple[str, str]], Dict[str, List[Tuple[str, str, str]]]
]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    refs_raw: List[str] = []
    hyps_raw: List[str] = []
    ids: List[str] = []
    idx_pairs: List[Tuple[str, str]] = []
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
            def _k(x: str) -> Tuple[int, Union[int, str]]:
                try:
                    return (0, int(x))
                except:
                    return (1, x)
            seq.sort(key=lambda t: _k(t[0]))
            groups[vid] = seq
    if not refs_raw:
        raise ValueError("No ground-truth captions found in JSON.")
    return refs_raw, hyps_raw, ids, idx_pairs, groups


def compute_bertscore(hyps: List[str], refs: List[str]) -> Tuple[float, float, float]:
    P, R, F1 = bert_score(hyps, refs, lang="en", rescale_with_baseline=True, verbose=False)
    return float(P.mean()), float(R.mean()), float(F1.mean())


def compute_sbert_sim(hyps: List[str], refs: List[str]) -> float:
    model = SentenceTransformer("all-mpnet-base-v2")
    e_h = model.encode(hyps, convert_to_tensor=True, normalize_embeddings=True)
    e_r = model.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
    sims = sbert_util.cos_sim(e_h, e_r).diagonal()
    return float(sims.mean().item())


def compute_nsp(groups: Dict[str, List[Tuple[str, str, str]]]) -> Tuple[float, float, float]:
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    mdl = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    mdl.eval()
    true_scores, shuf_scores = [], []
    for _, seq in groups.items():
        preds = [p for _, _, p in seq if p]
        if len(preds) < 2:
            continue
        for a, b in zip(preds[:-1], preds[1:]):
            enc = tok(a, b, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                probs = torch.softmax(mdl(**enc).logits, dim=1)[0]
            true_scores.append(float(probs[0].item()))
        import random
        shuf = preds[:]
        random.shuffle(shuf)
        for a, b in zip(shuf[:-1], shuf[1:]):
            enc = tok(a, b, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                probs = torch.softmax(mdl(**enc).logits, dim=1)[0]
            shuf_scores.append(float(probs[0].item()))
    if not true_scores or not shuf_scores:
        return float("nan"), float("nan"), float("nan")
    true_m = sum(true_scores) / len(true_scores)
    shuf_m = sum(shuf_scores) / len(shuf_scores)
    return true_m, shuf_m, (true_m - shuf_m)


def compute_dtw_alignment(groups: Dict[str, List[Tuple[str, str, str]]]) -> float:
    model = SentenceTransformer("all-mpnet-base-v2")
    dists = []
    for _, seq in groups.items():
        gts = [g for _, g, _ in seq if g]
        hyps = [p for _, _, p in seq if p]
        if not gts or not hyps:
            continue
        e_g = model.encode(gts, convert_to_tensor=False, normalize_embeddings=True)
        e_h = model.encode(hyps, convert_to_tensor=False, normalize_embeddings=True)
        def cos_dist(u, v):
            return 1.0 - float(sum(a*b for a, b in zip(u, v)))
        d, _ = fastdtw(e_h, e_g, dist=cos_dist)
        dists.append(float(d))
    return float(sum(dists) / len(dists)) if dists else float("nan")


def compute_nli_contradiction_rate(groups: Dict[str, List[Tuple[str, str, str]]]) -> float:
    tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
    mdl = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    mdl.eval()
    total, contrad = 0, 0
    for _, seq in groups.items():
        preds = [p for _, _, p in seq if p]
        for a, b in zip(preds[:-1], preds[1:]):
            enc = tok(a, b, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                logits = mdl(**enc).logits[0]
            label = int(torch.argmax(logits).item())  # 0=entail, 1=neutral, 2=contradiction
            total += 1
            if label == 2:
                contrad += 1
    return float(contrad / total) if total else float("nan")


def main():
    if len(sys.argv) < 2:
        print("Usage: python semanticMetrics.py /path/to/captions.json")
        sys.exit(1)
    json_path = sys.argv[1]
    refs_raw, hyps_raw, _, _, groups = load_pairs_from_json(json_path)

    p_mean, r_mean, f_mean = compute_bertscore(hyps_raw, refs_raw)
    sbert_sim = compute_sbert_sim(hyps_raw, refs_raw)
    nsp_true, nsp_shuf, nsp_delta = compute_nsp(groups)
    dtw_align = compute_dtw_alignment(groups)
    nli_contrad = compute_nli_contradiction_rate(groups)

    metrics: Dict[str, Any] = {
        "num_samples": len(refs_raw),
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
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
