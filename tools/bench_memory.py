"""
Memory/speed benchmark for the clone path, run outside the studio server.

    VOICEFORGE_MLX_CACHE_GB=64 .venv/bin/python tools/bench_memory.py   # old behaviour (cache ~unbounded)
    .venv/bin/python tools/bench_memory.py                              # new default (2 GB cache cap)

Renders the same lines with the same seeds for a locked character, printing per-line time and memory.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VOICEFORGE_IDLE_UNLOAD_MIN", "0")

from engine import VoiceForgeEngine, memory_snapshot, MLX_CACHE_LIMIT_GB  # noqa: E402

LINES = [
    "Hey, you made it. I honestly didn't think you'd come.",
    "No... no, that's not what I meant, and you know it!",
    "Fine. Let's just get this over with before anyone notices we're gone.",
    "Wait, did you hear that? Something's moving behind the door.",
    "I told you a hundred times, never, ever open that box!",
    "Okay. Okay, breathe. We can still fix this, I promise.",
]


def rss_gb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return round(int(out.strip() or 0) * 1024 / 1e9, 2)


def main():
    rounds = int(os.environ.get("BENCH_ROUNDS", "2"))
    chars = json.loads((ROOT / "data" / "characters.json").read_text())
    want = os.environ.get("BENCH_CHARACTER", "").lower()
    char = next((c for c in chars if want and want in c["name"].lower()), None) or next(
        c for c in chars if Path(c.get("file_path", "")).exists() and c.get("ref_text"))
    print(f"cache limit {MLX_CACHE_LIMIT_GB} GB | character {char['name']} | {rounds} rounds x {len(LINES)} lines", flush=True)

    eng = VoiceForgeEngine()
    t0 = time.perf_counter()
    eng.load_model("qwen3_clone")
    print(f"load {time.perf_counter() - t0:.1f}s  rss {rss_gb()} GB  {memory_snapshot()}", flush=True)

    prev_path, prev_text, total_audio, max_rss = None, None, 0.0, 0.0
    t_all = time.perf_counter()
    for r in range(rounds):
        for i, line in enumerate(LINES):
            audio, sr, m = eng.synthesize(
                "qwen3_clone", line, ref_audio_path=char["file_path"], ref_text=char["ref_text"],
                seed=1000 + i, prev_audio_path=prev_path, prev_text=prev_text,
            )
            out = Path("/tmp") / f"vf_bench_{i}.wav"
            import soundfile as sf
            sf.write(str(out), audio, sr)
            prev_path, prev_text = str(out), line
            total_audio += m["duration_sec"]
            rss = rss_gb()
            max_rss = max(max_rss, rss)
            snap = memory_snapshot()
            print(f"r{r} l{i}: {m['elapsed_sec']:5.1f}s for {m['duration_sec']:4.1f}s audio | rss {rss:5.2f} GB | "
                  f"active {snap['active_memory_gb']} cache {snap['cache_memory_gb']} line-peak {m['peak_memory_gb']}", flush=True)
    wall = time.perf_counter() - t_all
    print(f"TOTAL {wall:.1f}s for {total_audio:.1f}s audio (rtf {total_audio / wall:.2f}) | max rss {max_rss} GB | "
          f"after {memory_snapshot()}", flush=True)


if __name__ == "__main__":
    main()
