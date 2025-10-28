import sys
import json
import re
import string
import unicodedata
from typing import List, Tuple

import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider

def normalize(s: str) -> str:
    _PUNCT_TABLE = str.maketrans("", "", string.punctuation)
    s = unicodedata.normalize("NFKC", str(s)).lower().strip().translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s)
    return s

def load_pairs_from_json(path: str) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    refs_raw: List[str] = []
    hyps_raw: List[str] = []
    ids: List[str] = []
    idx_pairs: List[Tuple[str, str]] = []  # (video_id, scene_index)

    for vid, scenes in data.items():
        for sid, obj in scenes.items():
            gt = (obj.get("ground_truth") or "").strip()
            pr = (obj.get("predicted") or "").strip()
            # keep rows where at least GT exists
            if gt == "":
                continue
            refs_raw.append(gt)
            hyps_raw.append(pr)
            ids.append(f"{vid}|{sid}")
            idx_pairs.append((vid, sid))

    if not refs_raw:
        raise ValueError("No ground-truth captions found in JSON.")
    return refs_raw, hyps_raw, ids, idx_pairs

def main():
    if len(sys.argv) < 2:
        print("Usage: python metrics.py /path/to/captions.json")
        sys.exit(1)

    json_path = sys.argv[1]
    refs_raw, hyps_raw, ids, _ = load_pairs_from_json(json_path)

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

    metrics = {
        "num_samples": len(refs_raw),
        "BLEU4": bleu4,
        "METEOR": meteor,
        "CIDEr": float(cider_mean),
    }

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
