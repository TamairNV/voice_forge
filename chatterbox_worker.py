"""
Chatterbox TTS worker for VoiceForge Studio.

Chatterbox runs on PyTorch (MPS), which conflicts with the studio's MLX venv, so it lives in its own
venv (.venv-chatterbox) and the studio talks to this script over stdin/stdout, one JSON object per line.

Request:  {"text": "...", "out_path": "/tmp/x.wav", "audio_prompt_path": "/ref.wav" | null,
           "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "top_p": 1.0,
           "repetition_penalty": 1.2, "min_p": 0.05, "seed": 123 | null}
Response: {"ok": true, "sample_rate": 24000, "elapsed_sec": 1.2} or {"ok": false, "error": "..."}

The first line written is {"ready": true, ...} once the model has loaded.
Run: .venv-chatterbox/bin/python chatterbox_worker.py
"""

import gc
import json
import sys
import time
import traceback

# Anything the libraries print must not land on stdout, which carries the protocol
_proto = sys.stdout
sys.stdout = sys.stderr


def _send(obj):
    _proto.write(json.dumps(obj) + "\n")
    _proto.flush()


def _free(torch, device):
    gc.collect()
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device == "cuda":
        torch.cuda.empty_cache()


def main():
    t0 = time.perf_counter()
    try:
        import torch
        import torchaudio
        from chatterbox.tts import ChatterboxTTS

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        model = ChatterboxTTS.from_pretrained(device=device)
    except Exception as e:
        _send({"ready": False, "error": f"{type(e).__name__}: {e}"})
        return 1

    _free(torch, device)
    _send({"ready": True, "device": device, "sample_rate": model.sr, "load_sec": round(time.perf_counter() - t0, 2)})

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            if req.get("cmd") == "quit":
                break
            t1 = time.perf_counter()
            if req.get("seed") is not None:
                torch.manual_seed(int(req["seed"]))
            kwargs = dict(
                exaggeration=float(req.get("exaggeration", 0.5)),
                cfg_weight=float(req.get("cfg_weight", 0.5)),
                temperature=float(req.get("temperature", 0.8)),
                top_p=float(req.get("top_p", 1.0)),
                repetition_penalty=float(req.get("repetition_penalty", 1.2)),
                min_p=float(req.get("min_p", 0.05)),
            )
            prompt = req.get("audio_prompt_path")
            if prompt:
                kwargs["audio_prompt_path"] = prompt
            elif model.conds is None:
                raise ValueError("Chatterbox needs a reference voice for the first line (pick a locked voice).")
            wav = model.generate(req["text"], **kwargs)
            torchaudio.save(req["out_path"], wav.cpu(), model.sr)
            del wav
            _send({"ok": True, "sample_rate": model.sr, "elapsed_sec": round(time.perf_counter() - t1, 2)})
        except Exception as e:
            traceback.print_exc()
            _send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            # Without this the MPS cache balloons to ~30 GB over a long scene
            _free(torch, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
