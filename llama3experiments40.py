import os, io, json, glob, time, pathlib, re, base64, tempfile
from typing import List, Dict
from PIL import Image
import torch
os.environ['HF_HOME'] = '/workspace/hf'
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.environ['HF_HOME'], "transformers")
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.environ['HF_HOME'], "datasets")
from transformers import AutoModelForCausalLM, AutoProcessor

baseDir = "/workspace/scene_captioner/data/YouCookII/YouCookII/raw_videos/validation"
sampleFile = "/workspace/sampled_videos.txt"
gtFile = "/workspace/scene_captioner/data/YouCookII/YouCookII/annotations/youcookii_annotations_trainval.json"
output = "/workspace/llama3Captions40Round3.json"

PROMPT = (
    "You are given multiple frames from a short cooking clip, in chronological order. "
    "Write ONE concise sentence that is both descriptive, short, and instructional. "
    "Use an imperative tone, as if giving instructions for cooking or performing a task. "
    "Your response MUST be enclosed between <ANSWER> and </ANSWER>, containing ONLY the final instruction sentence."
)

STRIDE = 8
MAXFRAMES = 40
VIDEOLLAMA_NAME = "DAMO-NLP-SG/VideoLLaMA3-7B"

print(f"[start] device={'cuda' if torch.cuda.is_available() else 'cpu'} | model={VIDEOLLAMA_NAME} | baseDir={baseDir}", flush=True)
os.makedirs("offload", exist_ok=True)

_device = "cuda" if torch.cuda.is_available() else "cpu"
t0_load = time.time()
print("[stage] loading VideoLLaMA3 weights...", flush=True)
_model = AutoModelForCausalLM.from_pretrained(
    VIDEOLLAMA_NAME,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=(torch.float16 if _device == "cuda" else torch.float32),
    low_cpu_mem_usage=True,
    offload_folder="offload"
)
_processor = AutoProcessor.from_pretrained(VIDEOLLAMA_NAME, trust_remote_code=True)
print(f"[stage] model ready in {time.time()-t0_load:.1f}s", flush=True)

def read(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    print(f"[data] loaded {len(lines)} entries from {path}", flush=True)
    return lines

def getGT(gt_path: str) -> Dict[str, Dict[int, str]]:
    print(f"[stage] loading ground truth from {gt_path} ...", flush=True)
    with open(gt_path, "r", encoding="utf-8") as f:
        db = json.load(f).get("database", {})
    gt: Dict[str, Dict[int, str]] = {}
    for vid, meta in db.items():
        inner: Dict[int, str] = {}
        for a in meta.get("annotations", []):
            sid = int(a.get("id"))
            inner[sid] = a.get("sentence", "").strip()
        gt[vid] = inner
    print(f"[stage] ground truth loaded for {len(gt)} videos", flush=True)
    return gt

def findFolders(video_root: str) -> List[str]:
    if not os.path.isdir(video_root):
        print(f"[warn] video root missing: {video_root}", flush=True)
        return []
    subs = [p for p in glob.glob(os.path.join(video_root, "*")) if os.path.isdir(p)]
    keep = []
    for s in subs:
        if glob.glob(os.path.join(s, "*.jpg")) or glob.glob(os.path.join(s, "*.jpeg")):
            keep.append(s)
    def natural(p):
        name = os.path.basename(p)
        m = re.fullmatch(r"\d+", name)
        return (0, int(name)) if m else (1, name)
    keep_sorted = sorted(keep, key=natural)
    print(f"[scan] {video_root} -> {len(keep_sorted)} scene folders", flush=True)
    return keep_sorted

def sampleFrames(frameFolder: str, stride: int = STRIDE, limit: int = MAXFRAMES) -> List[str]:
    jpgs = sorted(
        glob.glob(os.path.join(frameFolder, "*.jpg")) +
        glob.glob(os.path.join(frameFolder, "*.jpeg"))
    )
    if not jpgs:
        print(f"[warn] no frames in {frameFolder}", flush=True)
        return []
    sampled = jpgs[::max(1, stride)]
    if len(sampled) > limit:
        idxs = [0]
        mids = max(0, limit - 2)
        if mids:
            step = max(1, (len(sampled) - 2) // mids)
            idxs += list(range(1, len(sampled) - 1, step))[:mids]
        idxs += [len(sampled) - 1]
        sampled = [sampled[i] for i in idxs]
    print(f"[frames] {frameFolder} -> using {len(sampled)} / {len(jpgs)} frames", flush=True)
    return sampled

def b64_from_path(path: str, maxSide: int = 768) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(w, h) / float(maxSide)
    if scale > 1.0:
        img = img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)
    bio = io.BytesIO()
    img.save(bio, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(bio.getvalue()).decode("utf-8")

def caption_with_videollama_from_paths(paths: List[str]) -> str:
    print(f"[infer] preparing {len(paths)} frames", flush=True)
    content = [{"type": "image", "image": {"image_path": p}} for p in paths]
    content.append({"type": "text", "text": PROMPT})
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]
    tprep = time.time()
    inputs = _processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    tensor_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            tensor_inputs[k] = v.to(_device)
        else:
            tensor_inputs[k] = v
    if "pixel_values" in tensor_inputs and tensor_inputs["pixel_values"].dtype != torch.float16 and _device == "cuda":
        tensor_inputs["pixel_values"] = tensor_inputs["pixel_values"].to(torch.float16)
    print(f"[infer] inputs ready in {time.time()-tprep:.1f}s; generating...", flush=True)
    tgen = time.time()
    with torch.inference_mode():
        output_ids = _model.generate(
            **tensor_inputs,
            max_new_tokens=60,
            do_sample=True,       
            top_p=0.9,            
            temperature=0.7       
        )
    text = _processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f"[infer] generation done in {time.time()-tgen:.1f}s", flush=True)
    m = re.search(r"<ANSWER>(.*?)</ANSWER>", text, flags=re.IGNORECASE | re.DOTALL)
    pred = (m.group(1) if m else text).strip()
    print(f"[infer] caption: {pred}", flush=True)
    return pred

def main():
    run_start = time.time()
    gt = getGT(gtFile)
    merged: Dict[str, Dict[str, Dict[str, str]]] = {}
    lines = read(sampleFile)
    processed = 0

    for rel in lines:
        videoId = pathlib.Path(rel).parts[-1]
        videoRoot = os.path.join(baseDir, rel)
        print(f"[video] {videoId} -> {videoRoot}", flush=True)

        sceneFolders = findFolders(videoRoot)
        if not sceneFolders:
            continue

        merged.setdefault(videoId, {})

        for idx, scenePath in enumerate(sceneFolders):
            t0 = time.time()
            folderName = os.path.basename(scenePath)
            sceneIndex = int(folderName) if folderName.isdigit() else idx
            print(f"[scene] {videoId}/{sceneIndex} sampling from {scenePath}", flush=True)

            framePaths = sampleFrames(scenePath)
            if not framePaths:
                print(f"[scene] {videoId}/{sceneIndex} skipped (no frames)", flush=True)
                continue

            try:
                pred_raw = caption_with_videollama_from_paths(framePaths)
                print(f"[scene] {videoId}/{sceneIndex} caption received", flush=True)
            except Exception as e:
                print(f"[error] {videoId}/{sceneIndex} inference failed: {e}", flush=True)
                time.sleep(0.25)
                try:
                    pred_raw = caption_with_videollama_from_paths(framePaths)
                    print(f"[scene] {videoId}/{sceneIndex} caption received (retry)", flush=True)
                except Exception as e2:
                    print(f"[error] {videoId}/{sceneIndex} retry failed: {e2}", flush=True)
                    pred_raw = ""

            gtSentence = gt.get(videoId, {}).get(sceneIndex, "")
            merged[videoId][str(sceneIndex)] = {"ground_truth": gtSentence, "predicted": pred_raw}

            with open(output, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)

            dt = time.time() - t0
            processed += 1
            avg = (time.time() - run_start) / processed if processed else 0.0
            print(f"[scene] {videoId}/{sceneIndex} done in {dt:.1f}s | total={processed} | avg={avg:.1f}s/scene | wrote {output}", flush=True)
            time.sleep(0.1)

    total = time.time() - run_start
    print(f"[done] wrote {output} | scenes={processed} | total={total:.1f}s | avg={(total/processed) if processed else 0.0:.1f}s/scene", flush=True)

if __name__ == "__main__":
    main()
