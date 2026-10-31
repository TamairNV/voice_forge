"""
Chatterbox TTS worker for VoiceForge Studio.

Chatterbox runs on PyTorch (MPS), which conflicts with the studio's MLX venv, so it lives in its own
venv (.venv-chatterbox) and the studio talks to this script over stdin/stdout, one JSON object per line.

Request:  {"text": "...", "out_path": "/tmp/x.wav", "audio_prompt_path": "/ref.wav" | null,
           "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "top_p": 1.0,
           "repetition_penalty": 1.2, "min_p": 0.05, "seed": 123 | null,
           "anchor_path": "/neutral.wav" | null, "emotion_blend": 0.35}
Response: {"ok": true, "sample_rate": 24000, "elapsed_sec": 1.2, "speaker_similarity": 0.91 | null}
          or {"ok": false, "error": "..."}

Anchor + emotion blend (Clone Studio profiles): with `anchor_path` set, the voice is built from the profile's neutral
clip (the anchor) and the line's emotion clip (`audio_prompt_path`), mixed by `emotion_blend` (0 = anchor only,
1 = emotion clip only). The delivery prompt (pace, mood) always comes from the emotion clip; the speaker fingerprint
is a weighted mix; the timbre reference is `emotion_blend` of the window from the emotion clip, then the anchor.
`speaker_similarity` is a Resemblyzer voice match (cosine) between the take and the anchor (or the reference clip),
used by the studio to pick the steadiest of N takes.

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


def _footprint_gb():
    """This process's memory as Activity Monitor counts it (macOS phys_footprint), or None elsewhere."""
    try:
        import ctypes
        import ctypes.util

        class RUsageInfoV2(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(25)]

        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        info = RUsageInfoV2()
        import os
        if libc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(info)) != 0:  # RUSAGE_INFO_V2
            return None
        return round(info.f7 / 1e9, 2)  # ri_phys_footprint
    except Exception:
        return None


def _free(torch, device):
    gc.collect()
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device == "cuda":
        torch.cuda.empty_cache()


BLEND_WINDOW_SEC = 8.0


class Blender:
    """Builds and caches Chatterbox conditionals for anchor + emotion blends, and scores takes against the anchor."""

    def __init__(self, model, torch):
        self.model, self.torch = model, torch
        self._conds = {}
        self._embs = {}
        try:
            from resemblyzer import VoiceEncoder
            self.encoder = VoiceEncoder("cpu", verbose=False)
        except Exception as e:  # scoring is optional; blending still works
            print(f"[chatterbox_worker] Resemblyzer unavailable, no speaker similarity: {e}", flush=True)
            self.encoder = None

    @staticmethod
    def _key(path):
        import os
        return (path, os.path.getmtime(path))  # a re-recorded clip at the same path must not hit the cache

    def _plain(self, path):
        key = self._key(path)
        if key not in self._conds:
            self.model.prepare_conditionals(path)
            self._conds[key] = self.model.conds
        return self._conds[key]

    def conds(self, emo_path, anchor_path, w):
        import librosa
        import numpy as np
        from chatterbox.tts import Conditionals, S3GEN_SR
        from chatterbox.models.t3.modules.cond_enc import T3Cond

        key = (self._key(emo_path), self._key(anchor_path), round(float(w), 3))
        if key in self._conds:
            return self._conds[key]
        a, e = self._plain(anchor_path), self._plain(emo_path)
        emb = (1 - w) * a.t3.speaker_emb + w * e.t3.speaker_emb
        emb = emb / emb.norm() * a.t3.speaker_emb.norm()
        emo_wav = librosa.load(emo_path, sr=S3GEN_SR)[0]
        anchor_wav = librosa.load(anchor_path, sr=S3GEN_SR)[0]
        # anchor last so it pulls harder
        mix = np.concatenate([emo_wav[: int(w * BLEND_WINDOW_SEC * S3GEN_SR)],
                              anchor_wav[: int((1 - w) * BLEND_WINDOW_SEC * S3GEN_SR)]])
        gen = self.model.s3gen.embed_ref(mix, S3GEN_SR, device=self.model.device)
        t3 = T3Cond(speaker_emb=emb, cond_prompt_speech_tokens=e.t3.cond_prompt_speech_tokens,
                    emotion_adv=0.5 * self.torch.ones(1, 1, 1)).to(device=self.model.device)
        self._conds[key] = Conditionals(t3, gen)
        return self._conds[key]

    def _embed(self, wav, sr):
        import librosa
        from resemblyzer import preprocess_wav
        return self.encoder.embed_utterance(preprocess_wav(librosa.resample(wav, orig_sr=sr, target_sr=16000), source_sr=16000))

    def similarity(self, wav, sr, ref_path):
        if self.encoder is None or not ref_path:
            return None
        import librosa
        import numpy as np
        key = self._key(ref_path)
        if key not in self._embs:
            self._embs[key] = self._embed(librosa.load(ref_path, sr=16000)[0], 16000)
        return round(float(np.dot(self._embed(wav, sr), self._embs[key])), 4)


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

    blender = Blender(model, torch)
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
            anchor = req.get("anchor_path")
            if prompt and anchor and anchor != prompt:
                from chatterbox.tts import Conditionals
                c = blender.conds(prompt, anchor, max(0.0, min(1.0, float(req.get("emotion_blend", 0.35)))))
                # generate() swaps in a new T3Cond for the exaggeration, so hand it a fresh wrapper each time
                model.conds = Conditionals(c.t3, c.gen)
            elif prompt:
                kwargs["audio_prompt_path"] = prompt
            elif model.conds is None:
                raise ValueError("Chatterbox needs a reference voice for the first line (pick a locked voice).")
            wav = model.generate(req["text"], **kwargs)
            torchaudio.save(req["out_path"], wav.cpu(), model.sr)
            sim = blender.similarity(wav.squeeze().cpu().numpy(), model.sr, anchor or prompt)
            del wav
            _free(torch, device)
            _send({"ok": True, "sample_rate": model.sr, "elapsed_sec": round(time.perf_counter() - t1, 2),
                   "speaker_similarity": sim, "footprint_gb": _footprint_gb()})
        except Exception as e:
            traceback.print_exc()
            _send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            # Without this the MPS cache balloons to ~30 GB over a long scene
            _free(torch, device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
