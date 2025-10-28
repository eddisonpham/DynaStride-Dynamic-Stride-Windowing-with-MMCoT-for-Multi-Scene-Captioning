import os, io, json, glob, time, pathlib, re, base64, random
from typing import List, Dict
from PIL import Image
import httpx
from openai import OpenAI
from openai._exceptions import OpenAIError
from httpx import HTTPStatusError, TimeoutException

baseDir    = "/workspace/scene_captioner/data/YouCookII/YouCookII/raw_videos/validation"
sampleFile = "/workspace/sampled_videos.txt"
gtFile     = "/workspace/scene_captioner/data/YouCookII/YouCookII/annotations/youcookii_annotations_trainval.json"
output     = "/workspace/gpt40Captions20Round2.json"

PROMPT = (
    "You are given multiple frames from a short cooking clip, in chronological order. "
    "Write ONE concise sentence that is both descriptive and instructional. "
    "Use an imperative tone, as if giving step-by-step directions for cooking or performing a task. "
    "Your response MUST be enclosed between <ANSWER> and </ANSWER>, containing ONLY the final instruction sentence."
)

STRIDE = int(os.environ.get("STRIDE", "5"))
MAXFRAMES = int(os.environ.get("MAXFRAMES", "20"))
STRIP_GAP = int(os.environ.get("STRIP_GAP", "4"))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_REQ_DELAY = float(os.environ.get("OPENAI_REQ_DELAY", "0.7"))
MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.environ.get("OPENAI_BACKOFF_BASE", "0.8"))
BACKOFF_MAX = float(os.environ.get("OPENAI_BACKOFF_MAX", "15"))

RETRIABLE_STATUS = {408, 413, 429, 500, 502, 503, 504}

client = OpenAI(max_retries=0, timeout=httpx.Timeout(connect=8.0, read=20.0, write=20.0, pool=8.0))

def read(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

def getGT(gt_path: str) -> Dict[str, Dict[int, str]]:
    with open(gt_path, "r", encoding="utf-8") as f:
        db = json.load(f).get("database", {})
    gt: Dict[str, Dict[int, str]] = {}
    for vid, meta in db.items():
        inner: Dict[int, str] = {}
        for a in meta.get("annotations", []):
            sid = int(a.get("id"))
            inner[sid] = a.get("sentence", "").strip()
        gt[vid] = inner
    return gt

def findFolders(video_root: str) -> List[str]:
    if not os.path.isdir(video_root):
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
    return sorted(keep, key=natural)

def sampleFrames(frameFolder: str, stride: int = STRIDE, limit: int = MAXFRAMES) -> List[str]:
    jpgs = sorted(
        glob.glob(os.path.join(frameFolder, "*.jpg")) +
        glob.glob(os.path.join(frameFolder, "*.jpeg"))
    )
    if not jpgs:
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
    return sampled

def evenly_sample(items: List[str], k: int) -> List[str]:
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return items
    step = n / float(k)
    return [items[int(i * step)] for i in range(k)]

def b64_size_bytes(data_url: str) -> int:
    b64 = data_url.split(",", 1)[1]
    return (len(b64) * 3) // 4

def make_horizontal_strip_data_url(paths: List[str], max_height: int, max_frames: int, gap: int, quality: int) -> str:
    print(f"[strip] building strip: frames={max_frames} height={max_height} gap={gap} q={quality}", flush=True)
    chosen = evenly_sample(paths, min(max_frames, len(paths)))
    frames = []
    for p in chosen:
        try:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            if h != max_height:
                new_w = max(1, int(w * (max_height / float(h))))
                img = img.resize((new_w, max_height), Image.LANCZOS)
            frames.append(img)
        except Exception as e:
            print(f"[strip] skip frame {p}: {e}", flush=True)
            continue
    if not frames:
        raise RuntimeError("no frames available to build strip")
    total_w = sum(im.width for im in frames) + gap * (len(frames) - 1)
    strip = Image.new("RGB", (total_w, max_height), (255, 255, 255))
    x = 0
    for i, im in enumerate(frames):
        strip.paste(im, (x, 0))
        x += im.width + (gap if i < len(frames) - 1 else 0)
    bio = io.BytesIO()
    strip.save(bio, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
    url = f"data:image/jpeg;base64,{b64}"
    print(f"[strip] built strip {total_w}x{max_height}, payload≈{b64_size_bytes(url)/1024:.1f}KB", flush=True)
    return url

def _sleep_backoff(attempt: int, retry_after_header: str | None):
    if retry_after_header:
        try:
            sec = float(retry_after_header)
            print(f"[retry] server retry-after={sec}s", flush=True)
            time.sleep(min(sec, BACKOFF_MAX))
            return
        except Exception:
            pass
    delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** attempt)) + random.uniform(0, 0.4)
    print(f"[retry] backoff {delay:.2f}s (attempt {attempt+1})", flush=True)
    time.sleep(delay)

def caption_with_openai_from_paths(paths: List[str]) -> str:
    profiles = [
        (6, 256, 4, 72, 18.0),
        (5, 224, 4, 68, 16.0),
        (4, 192, 4, 64, 14.0),
        (3, 160, 4, 60, 12.0),
    ]
    last_err = None
    for frames, height, gap, quality, read_timeout in profiles:
        try:
            strip_url = make_horizontal_strip_data_url(paths, height, frames, gap, quality)
        except Exception as e:
            last_err = e
            print(f"[strip] failed to build: {e}", flush=True)
            continue
        tm = httpx.Timeout(connect=8.0, read=read_timeout, write=read_timeout, pool=8.0)
        local_client = client.with_options(timeout=tm)
        content = [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": strip_url}},
        ]
        for attempt in range(MAX_RETRIES):
            try:
                print(f"[openai] request start model={OPENAI_MODEL} timeout={read_timeout}s attempt={attempt+1}", flush=True)
                t0 = time.time()
                resp = local_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.2,
                    max_tokens=120,
                )
                dt = time.time() - t0
                print(f"[openai] request ok in {dt:.1f}s", flush=True)
                text = resp.choices[0].message.content
                m = re.search(r"<ANSWER>(.*?)</ANSWER>", text, flags=re.IGNORECASE | re.DOTALL)
                return (m.group(1) if m else text).strip()
            except HTTPStatusError as e:
                last_err = e
                status = e.response.status_code if e.response is not None else None
                print(f"[openai] http {status} on attempt {attempt+1}", flush=True)
                if status in RETRIABLE_STATUS:
                    ra = e.response.headers.get("retry-after") if e.response is not None else None
                    _sleep_backoff(attempt, ra)
                    continue
                raise
            except (TimeoutException, OpenAIError, Exception) as e:
                last_err = e
                print(f"[openai] error on attempt {attempt+1}: {e}", flush=True)
                _sleep_backoff(attempt, None)
                continue
    if last_err:
        raise last_err
    raise RuntimeError("OpenAI request failed without exception")

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    print(f"[start] model={OPENAI_MODEL} baseDir={baseDir}", flush=True)
    print("[stage] loading ground truth...", flush=True)
    gt = getGT(gtFile)
    print(f"[stage] ground truth loaded: {len(gt)} videos", flush=True)
    lines = read(sampleFile)
    print(f"[stage] samples loaded: {len(lines)} entries from {sampleFile}", flush=True)
    merged: Dict[str, Dict[str, Dict[str, str]]] = {}
    processed = 0
    started = time.time()
    for rel in lines:
        videoId = pathlib.Path(rel).parts[-1]
        videoRoot = os.path.join(baseDir, rel)
        print(f"[video] {videoId} scanning scenes in {videoRoot}", flush=True)
        sceneFolders = findFolders(videoRoot)
        print(f"[video] {videoId} found {len(sceneFolders)} scene folders", flush=True)
        if not sceneFolders:
            continue
        merged.setdefault(videoId, {})
        for idx, scenePath in enumerate(sceneFolders):
            t0 = time.time()
            folderName = os.path.basename(scenePath)
            sceneIndex = int(folderName) if folderName.isdigit() else idx
            print(f"[scene] {videoId}/{sceneIndex} sampling frames from {scenePath}", flush=True)
            framePaths = sampleFrames(scenePath)
            print(f"[scene] {videoId}/{sceneIndex} sampled {len(framePaths)} frames", flush=True)
            if not framePaths:
                continue
            try:
                pred_raw = caption_with_openai_from_paths(framePaths)
                print(f"[scene] {videoId}/{sceneIndex} caption received", flush=True)
            except Exception as e:
                print(f"[scene] {videoId}/{sceneIndex} failed: {e}", flush=True)
                pred_raw = ""
            gtSentence = gt.get(videoId, {}).get(sceneIndex, "")
            merged[videoId][str(sceneIndex)] = {"ground_truth": gtSentence, "predicted": pred_raw}
            with open(output, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            dt = time.time() - t0
            processed += 1
            elapsed = time.time() - started
            avg = elapsed / processed if processed else 0.0
            print(f"[scene] {videoId}/{sceneIndex} done in {dt:.1f}s | total={processed} | avg={avg:.1f}s/scene | wrote {output}", flush=True)
            time.sleep(OPENAI_REQ_DELAY)
    total = time.time() - started
    print(f"[done] wrote {output} | scenes={processed} | total={total:.1f}s | avg={ (total/processed) if processed else 0.0:.1f}s/scene", flush=True)

if __name__ == "__main__":
    main()
