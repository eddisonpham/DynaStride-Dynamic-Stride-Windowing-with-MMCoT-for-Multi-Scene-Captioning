"""GPT (OpenAI) model wrapper for video captioning."""
import os
import re
import time
import random
from typing import List
import httpx
from openai import OpenAI
from openai._exceptions import OpenAIError
from httpx import HTTPStatusError, TimeoutException
from dotenv import load_dotenv

import sys
ROOT_DIR=os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
print(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
load_dotenv(dotenv_path = os.path.join(ROOT_DIR, "src/experiments/.env") )


from src.utils.image_processing import make_horizontal_strip_data_url


PROMPT_GPT = (
    "You are given multiple frames from a short cooking clip, in chronological order. "
    "Write ONE concise sentence that is both descriptive and instructional. "
    "Use an imperative tone, as if giving step-by-step directions for cooking or performing a task. "
    "Your response MUST be enclosed between <ANSWER> and </ANSWER>, containing ONLY the final instruction sentence."
)

STRIDE = 5
STRIP_GAP = 4
OPENAI_MODEL = "gpt-4o"
OPENAI_REQ_DELAY = 0.7
MAX_RETRIES = 5
BACKOFF_BASE = 0.8
BACKOFF_MAX = 15.0
GPT_KEY = os.getenv("OPEN_AI_API_KEY")

RETRIABLE_STATUS = {408, 413, 429, 500, 502, 503, 504}

_client = None


def get_openai_client():
    """Get or create OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=GPT_KEY,
            max_retries=0,
            timeout=httpx.Timeout(connect=8.0, read=20.0, write=20.0, pool=8.0)
        )
    return _client


def _sleep_backoff(attempt: int, retry_after_header: str | None):
    """Sleep with exponential backoff."""
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


def caption_with_openai(paths: List[str], max_frames: int = 10) -> str:
    """Generate caption from frame paths using OpenAI GPT."""
    profiles = [
        (6, 256, 4, 72, 18.0),
        (5, 224, 4, 68, 16.0),
        (4, 192, 4, 64, 14.0),
        (3, 160, 4, 60, 12.0),
    ]
    last_err = None
    client = get_openai_client()
    
    for frames, height, gap, quality, read_timeout in profiles:
        try:
            strip_url = make_horizontal_strip_data_url(paths, height, min(max_frames, frames), gap, quality)
        except Exception as e:
            last_err = e
            print(f"[strip] failed to build: {e}", flush=True)
            continue
        tm = httpx.Timeout(connect=8.0, read=read_timeout, write=read_timeout, pool=8.0)
        local_client = client.with_options(timeout=tm)
        content = [
            {"type": "text", "text": PROMPT_GPT},
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

