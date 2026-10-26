"""
VoiceForge Studio — Local Apple Silicon MLX Engine for Qwen3-TTS & Maya-1.
Supports hot-swapping between:
  1. Qwen3-TTS 1.7B VoiceDesign (bf16) — Design a voice from a natural language description
  2. Qwen3-TTS 1.7B Base (bf16)        — Zero-shot voice cloning from a golden reference audio clip
  3. Maya-1 3.3B (8-bit MLX + SNAC)    — Expressive character acting with inline emotion tags (<laugh>, <sigh>, etc.)
"""

import gc
import json
import subprocess
import tempfile
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer

from mlx_audio.tts.utils import load_model as load_mlx_tts_model
from mlx_audio.codec.models.snac import SNAC
from mlx_audio.lm.generate import stream_generate
from mlx_audio.lm.models.llama import Model as LlamaModel, ModelArgs as LlamaModelConfig
from mlx_audio.lm.sample_utils import make_logits_processors, make_sampler
from mlx_audio.utils import get_model_path, load_config, load_weights, apply_quantization, from_dict


def apply_studio_pacing(
    audio_np: np.ndarray,
    sr: int = 24000,
    speed: float = 1.0,
    trim_silence: bool = True,
) -> np.ndarray:
    """
    Studio post-processor:
    1. Tighten long awkward pauses (>260ms) down to snappy ~110ms conversational breaths.
    2. Apply pitch-preserving SOLA time-stretching via ffmpeg `atempo` when speed != 1.0.
    """
    if audio_np.size == 0:
        return audio_np

    # Step 1: Tighten dead-air pauses if enabled
    if trim_silence:
        win = int(sr * 0.02)  # 20ms frame
        if len(audio_np) > win * 10:
            n_frames = len(audio_np) // win
            frames = audio_np[: n_frames * win].reshape(n_frames, win)
            rms = np.sqrt(np.mean(frames ** 2, axis=1))
            thresh = max(0.004, float(np.percentile(rms, 15)) * 1.2)
            is_silent = rms < thresh

            max_silent_frames = 6  # keep at most 120ms of consecutive silence
            keep_chunks = []
            silent_run = 0
            for idx in range(n_frames):
                if is_silent[idx]:
                    silent_run += 1
                    if silent_run <= max_silent_frames:
                        keep_chunks.append(frames[idx])
                else:
                    silent_run = 0
                    keep_chunks.append(frames[idx])

            remainder = audio_np[n_frames * win :]
            if remainder.size > 0:
                keep_chunks.append(remainder)
            if keep_chunks:
                audio_np = np.concatenate(keep_chunks, axis=0)

    # Step 2: Pitch-preserving tempo adjustment via ffmpeg atempo
    if speed is not None and abs(speed - 1.0) >= 0.02:
        clamped_speed = max(0.5, min(2.0, float(speed)))
        with tempfile.TemporaryDirectory() as tmpdir:
            in_wav = Path(tmpdir) / "in.wav"
            out_wav = Path(tmpdir) / "out.wav"
            sf.write(str(in_wav), audio_np, sr, subtype="PCM_16")
            cmd = [
                "/opt/homebrew/bin/ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(in_wav),
                "-filter:a",
                f"atempo={clamped_speed}",
                str(out_wav),
            ]
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode == 0 and out_wav.exists():
                stretched, _ = sf.read(str(out_wav), dtype="float32")
                audio_np = stretched

    return audio_np

# Maya-1 Token Constants
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009

MODEL_REGISTRY = {
    "qwen3_voicedesign": {
        "id": "qwen3_voicedesign",
        "name": "Qwen3-TTS 1.7B — Voice Design (bf16)",
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "family": "qwen3",
        "mode": "voice_design",
        "vram_gb": 4.5,
        "badge": "Best for Inventing Voices",
        "description": "Create brand-new studio voices from a text description of the character, age, timbre, accent, and pacing.",
    },
    "maya1": {
        "id": "maya1",
        "name": "Maya-1 3.3B — Expressive Acting (MLX)",
        "repo": "nhe-ai/maya1-mlx-8Bit",
        "family": "maya1",
        "mode": "voice_design_tags",
        "vram_gb": 3.8,
        "badge": "Best for Inline Emotions",
        "description": "3.3B Llama-based expressive voice designer with 20+ inline acting tags (<laugh>, <sigh>, <whisper>, <gasp>, <cry>, <angry>).",
    },
    "qwen3_clone": {
        "id": "qwen3_clone",
        "name": "Qwen3-TTS 1.7B — Voice Lock / Clone (bf16)",
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "family": "qwen3",
        "mode": "voice_clone",
        "vram_gb": 4.5,
        "badge": "100% Consistent NPC Voices",
        "description": "Locks onto a saved Character Voice from your roster (zero-shot 3s–15s reference) so 1,000+ game dialogue lines never drift.",
    },
}

MAYA1_EMOTION_TAGS = [
    {"tag": "<laugh>", "label": "Laugh", "category": "Vocal"},
    {"tag": "<laugh_harder>", "label": "Hard Laugh", "category": "Vocal"},
    {"tag": "<chuckle>", "label": "Chuckle", "category": "Vocal"},
    {"tag": "<giggle>", "label": "Giggle", "category": "Vocal"},
    {"tag": "<sigh>", "label": "Sigh", "category": "Breath"},
    {"tag": "<exhale>", "label": "Exhale", "category": "Breath"},
    {"tag": "<gasp>", "label": "Gasp", "category": "Breath"},
    {"tag": "<gulp>", "label": "Gulp", "category": "Breath"},
    {"tag": "<snort>", "label": "Snort", "category": "Vocal"},
    {"tag": "<whisper>", "label": "Whisper", "category": "Style"},
    {"tag": "<angry>", "label": "Angry", "category": "Emotion"},
    {"tag": "<sarcastic>", "label": "Sarcastic", "category": "Emotion"},
    {"tag": "<mischievous>", "label": "Mischievous", "category": "Emotion"},
    {"tag": "<excited>", "label": "Excited", "category": "Emotion"},
    {"tag": "<curious>", "label": "Curious", "category": "Emotion"},
    {"tag": "<disappointed>", "label": "Disappointed", "category": "Emotion"},
    {"tag": "<appalled>", "label": "Appalled", "category": "Emotion"},
    {"tag": "<cry>", "label": "Cry", "category": "Emotion"},
    {"tag": "<scream>", "label": "Scream", "category": "Style"},
    {"tag": "<sing>", "label": "Sing", "category": "Style"},
]


class Maya1MLXRunner:
    """Native Apple Silicon MLX runner for Maya-1 (3.3B Llama + SNAC 24kHz)."""

    def __init__(self, repo_id: str = "nhe-ai/maya1-mlx-8Bit"):
        self.repo_id = repo_id
        self.sample_rate = 24000
        model_path = get_model_path(repo_id)
        config_dict = load_config(model_path)

        llama_args = from_dict(LlamaModelConfig, config_dict)
        self.model = LlamaModel(llama_args)

        weights = load_weights(model_path)
        if hasattr(self.model, "sanitize"):
            weights = self.model.sanitize(weights)

        apply_quantization(self.model, config_dict, weights)
        self.model.load_weights(list(weights.items()), strict=False)
        mx.eval(self.model.parameters())
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.snac = SNAC.from_pretrained("mlx-community/snac_24khz")
        self.snac.eval()

    def build_prompt(self, description: str, text: str) -> str:
        soh_token = self.tokenizer.decode([SOH_ID])
        eoh_token = self.tokenizer.decode([EOH_ID])
        soa_token = self.tokenizer.decode([SOA_ID])
        sos_token = self.tokenizer.decode([CODE_START_TOKEN_ID])
        eot_token = self.tokenizer.decode([TEXT_EOT_ID])
        bos_token = self.tokenizer.bos_token or "<|begin_of_text|>"

        clean_desc = description.replace('"', "'").strip()
        formatted_text = f'<description="{clean_desc}"> {text.strip()}'
        return soh_token + bos_token + formatted_text + eot_token + eoh_token + soa_token + sos_token

    @staticmethod
    def extract_snac_codes(token_ids: List[int]) -> List[int]:
        try:
            eos_idx = token_ids.index(CODE_END_TOKEN_ID)
        except ValueError:
            eos_idx = len(token_ids)
        return [
            tid for tid in token_ids[:eos_idx]
            if SNAC_MIN_ID <= tid <= SNAC_MAX_ID
        ]

    @staticmethod
    def unpack_snac_from_7(snac_tokens: List[int]) -> List[List[int]]:
        if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
            snac_tokens = snac_tokens[:-1]

        frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
        snac_tokens = snac_tokens[: frames * SNAC_TOKENS_PER_FRAME]
        if frames == 0:
            return [[], [], []]

        l1, l2, l3 = [], [], []
        for i in range(frames):
            slots = snac_tokens[i * 7 : (i + 1) * 7]
            l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
            l2.extend([
                (slots[1] - CODE_TOKEN_OFFSET) % 4096,
                (slots[4] - CODE_TOKEN_OFFSET) % 4096,
            ])
            l3.extend([
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ])
        return [l1, l2, l3]

    def generate(
        self,
        text: str,
        instruct: str,
        temperature: float = 0.45,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        max_tokens: int = 2048,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = time.perf_counter()
        prompt_str = self.build_prompt(instruct, text)
        encoded = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        prompt_ids = mx.array(encoded, dtype=mx.int32)

        sampler = make_sampler(temperature, top_p, top_k=50)
        logits_processors = make_logits_processors(
            logit_bias=None,
            repetition_penalty=repetition_penalty,
            repetition_context_size=64,
        )

        generated_ids: List[int] = []
        for resp in stream_generate(
            self.model,
            tokenizer=self.tokenizer,
            prompt=prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            tok = int(resp.token)
            if tok == CODE_END_TOKEN_ID and len(generated_ids) >= 28:
                break
            generated_ids.append(tok)
            if len(generated_ids) % 100 == 0:
                mx.clear_cache()

        snac_tokens = self.extract_snac_codes(generated_ids)
        levels = self.unpack_snac_from_7(snac_tokens)
        if not levels[0]:
            raise RuntimeError("Maya-1 did not produce enough acoustic tokens. Try adjusting temperature or description.")

        codes_mx = [mx.array(level, dtype=mx.int32)[None, :] for level in levels]
        audio_hat = self.snac.decode(codes_mx).squeeze()
        mx.eval(audio_hat)
        audio_np = np.array(audio_hat, dtype=np.float32)

        # Trim 2048-sample SNAC decoder warm-up transient if present
        if audio_np.shape[0] > 4096:
            audio_np = audio_np[2048:]

        elapsed = time.perf_counter() - t0
        duration = audio_np.shape[0] / self.sample_rate
        mx.clear_cache()

        return audio_np, {
            "duration_sec": round(duration, 2),
            "elapsed_sec": round(elapsed, 2),
            "rtf": round(duration / elapsed, 2) if elapsed > 0 else 0,
            "tokens": len(generated_ids),
            "frames": len(levels[0]),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        }


class VoiceForgeEngine:
    """Thread-safe model manager that hot-swaps between Qwen3-TTS and Maya-1 on Apple Silicon."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active_model_id: Optional[str] = None
        self.active_model: Any = None
        self.loading_model_id: Optional[str] = None
        self.status_message: str = "Ready to load model"

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_model_id": self.active_model_id,
            "loading_model_id": self.loading_model_id,
            "status_message": self.status_message,
            "active_memory_gb": round(mx.get_active_memory() / 1e9, 2),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
            "models": list(MODEL_REGISTRY.values()),
            "maya1_tags": MAYA1_EMOTION_TAGS,
        }

    def load_model(self, model_id: str) -> Dict[str, Any]:
        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_id: {model_id}")

        with self._lock:
            if self.active_model_id == model_id and self.active_model is not None:
                return self.get_status()

            info = MODEL_REGISTRY[model_id]
            self.loading_model_id = model_id
            self.status_message = f"Swapping to {info['name']}..."

            try:
                # Unload current model cleanly to keep Unified Memory low
                if self.active_model is not None:
                    del self.active_model
                    self.active_model = None
                    self.active_model_id = None
                    gc.collect()
                    mx.clear_cache()

                t0 = time.perf_counter()
                if info["family"] == "maya1":
                    self.active_model = Maya1MLXRunner(info["repo"])
                else:
                    self.active_model = load_mlx_tts_model(info["repo"])

                load_sec = time.perf_counter() - t0
                self.active_model_id = model_id
                self.loading_model_id = None
                self.status_message = f"Loaded {info['name']} in {load_sec:.1f}s"
                return self.get_status()
            except Exception as e:
                self.loading_model_id = None
                self.status_message = f"Error loading {model_id}: {e}"
                raise

    def synthesize(
        self,
        model_id: str,
        text: str,
        instruct: str = "",
        language: str = "english",
        speed: float = 1.0,
        trim_silence: bool = True,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        # Ensure requested model is active
        if self.active_model_id != model_id or self.active_model is None:
            self.load_model(model_id)

        with self._lock:
            info = MODEL_REGISTRY[model_id]
            t0 = time.perf_counter()

            if info["family"] == "maya1":
                temp = temperature if temperature is not None else 0.45
                tp = top_p if top_p is not None else 0.90
                rp = repetition_penalty if repetition_penalty is not None else 1.15
                audio_np, metrics = self.active_model.generate(
                    text=text,
                    instruct=instruct or "Realistic voice in their 30s, clear diction, natural conversational pacing.",
                    temperature=temp,
                    top_p=tp,
                    repetition_penalty=rp,
                )
                audio_np = apply_studio_pacing(audio_np, 24000, speed=speed, trim_silence=trim_silence)
                metrics["duration_sec"] = round(audio_np.shape[0] / 24000, 2)
                metrics["speed"] = round(speed, 2)
                return audio_np, 24000, metrics

            # Qwen3-TTS family
            temp = temperature if temperature is not None else 0.85
            tp = top_p if top_p is not None else 1.0
            rp = repetition_penalty if repetition_penalty is not None else 1.05

            if info["mode"] == "voice_design":
                if not instruct.strip():
                    raise ValueError("Please provide a character/voice description for Qwen3 VoiceDesign.")
                results = list(
                    self.active_model.generate_voice_design(
                        text=text,
                        instruct=instruct,
                        language=language,
                        temperature=temp,
                        top_p=tp,
                        repetition_penalty=rp,
                    )
                )
            elif info["mode"] == "voice_clone":
                if not ref_audio_path or not ref_text:
                    raise ValueError("Voice Lock / Clone mode requires selecting a saved character voice (reference audio + transcript).")
                results = list(
                    self.active_model.generate(
                        text=text,
                        ref_audio=ref_audio_path,
                        ref_text=ref_text,
                        lang_code=language,
                        temperature=temp,
                        top_p=tp,
                        repetition_penalty=rp,
                    )
                )
            else:
                raise ValueError(f"Unsupported mode: {info['mode']}")

            if not results:
                raise RuntimeError("Model returned no audio segments.")

            chunks = [np.array(r.audio, dtype=np.float32) for r in results if r.audio is not None]
            audio_np = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
            sr = self.active_model.sample_rate
            audio_np = apply_studio_pacing(audio_np, sr, speed=speed, trim_silence=trim_silence)
            elapsed = time.perf_counter() - t0
            duration = audio_np.shape[0] / sr
            total_tokens = sum(r.token_count for r in results)

            return audio_np, sr, {
                "duration_sec": round(duration, 2),
                "elapsed_sec": round(elapsed, 2),
                "rtf": round(duration / elapsed, 2) if elapsed > 0 else 0,
                "tokens": total_tokens,
                "speed": round(speed, 2),
                "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
            }
