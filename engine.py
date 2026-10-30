"""
VoiceForge Studio — Local Apple Silicon MLX Engine for Qwen3-TTS & Maya-1.
Supports hot-swapping between:
  1. Qwen3-TTS 1.7B VoiceDesign (bf16) — Design a voice from a natural language description
  2. Qwen3-TTS 1.7B Base (bf16)        — Zero-shot voice cloning from a golden reference audio clip
  3. Maya-1 3.3B (8-bit MLX + SNAC)    — Expressive character acting with inline emotion tags (<laugh>, <sigh>, etc.)
"""

import gc
import json
import re
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
from mlx_audio.utils import get_model_path, load_audio, load_config, load_weights, apply_quantization, from_dict


def normalize_studio_loudness(
    audio_np: np.ndarray,
    sr: int = 24000,
    target_rms: float = 0.14,
    peak_ceiling: float = 0.92,
) -> np.ndarray:
    """
    Studio loudness normalizer & micro-fade processor:
    - Removes DC offset
    - Normalizes active speech RMS to `target_rms` (~-17 dBFS) so Qwen3 and Maya-1 match in volume
    - Soft-limits peaks at `peak_ceiling` (-0.7 dBFS) so nothing clips
    - Applies 12ms cosine fade-in / fade-out to eliminate boundary clicks when stitching skits
    """
    if audio_np.size == 0:
        return audio_np

    audio = audio_np.astype(np.float32)
    audio = audio - float(np.mean(audio))

    # Measure active RMS (ignoring near-silent frames)
    abs_a = np.abs(audio)
    active = audio[abs_a > 0.008]
    ref = active if active.size > int(sr * 0.05) else audio
    rms = float(np.sqrt(np.mean(ref ** 2))) + 1e-8

    gain = target_rms / rms
    gain = min(gain, 6.0)  # Avoid over-amplifying quiet breaths
    audio = audio * gain

    # Soft tanh saturation above 0.75 up to peak_ceiling
    peak = float(np.max(np.abs(audio)))
    if peak > peak_ceiling:
        audio = np.tanh(audio / peak_ceiling) * peak_ceiling

    # 12ms fade-in and fade-out for click-free skit stitching
    fade_len = min(int(sr * 0.012), audio.shape[0] // 4)
    if fade_len > 1:
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        audio[:fade_len] *= fade_in
        audio[-fade_len:] *= fade_out

    return audio


def analyze_voice_acoustics(
    audio_np: np.ndarray,
    sr: int = 24000,
    spoken_text: str = "",
) -> Dict[str, Any]:
    """
    Read-only acoustic telemetry analyzer (does NOT modify audio):
      - Measures fundamental vocal pitch (F0 in Hz) across voiced frames via normalized autocorrelation (65Hz–450Hz)
      - Computes pitch median (Hz), 10th–90th percentile range (Hz), and intonation span (semitones)
      - Classifies vocal register and measures spoken cadence (words per minute)
    """
    if audio_np.size < int(sr * 0.08):
        return {
            "pitch_hz_median": 0.0,
            "pitch_hz_p10": 0.0,
            "pitch_hz_p90": 0.0,
            "pitch_semitone_span": 0.0,
            "voice_register": "Unknown",
            "rms_dbfs": -60.0,
            "words_per_min": 0,
        }

    audio = audio_np.astype(np.float32)
    duration_sec = max(0.01, audio.shape[0] / sr)

    active_samples = audio[np.abs(audio) > 0.01]
    rms_val = float(np.sqrt(np.mean((active_samples if active_samples.size > 100 else audio) ** 2))) + 1e-9
    rms_dbfs = round(20.0 * float(np.log10(rms_val)), 1)

    frame_len = int(sr * 0.04)
    hop_len = int(sr * 0.02)
    min_lag = int(sr / 450.0)
    max_lag = int(sr / 65.0)

    n_frames = max(0, (audio.shape[0] - frame_len) // hop_len + 1)
    f0_estimates: List[float] = []

    if n_frames > 0:
        window = np.hanning(frame_len).astype(np.float32)
        fft_len = 2048
        frame_energies = []
        frames_list = []
        for i in range(n_frames):
            f = audio[i * hop_len : i * hop_len + frame_len]
            frame_energies.append(float(np.sqrt(np.mean(f ** 2))))
            frames_list.append(f)

        energy_thresh = max(0.015, float(np.percentile(frame_energies, 35)))
        for f, e in zip(frames_list, frame_energies):
            if e < energy_thresh:
                continue
            fw = (f - float(np.mean(f))) * window
            zcr = float(np.mean(np.abs(np.diff(np.signbit(fw)))))
            if zcr > 0.22:
                continue

            spec = np.fft.rfft(fw, n=fft_len)
            acf = np.fft.irfft(spec * np.conj(spec), n=fft_len)[: max_lag + 2]
            if acf[0] <= 1e-8:
                continue
            acf_norm = acf / acf[0]
            search_region = acf_norm[min_lag : max_lag + 1]
            if search_region.size == 0:
                continue

            peak_idx = int(np.argmax(search_region))
            if float(search_region[peak_idx]) >= 0.48:
                lag = min_lag + peak_idx
                f0 = float(sr / lag)
                if 65.0 <= f0 <= 450.0:
                    f0_estimates.append(f0)

    if f0_estimates:
        f0_arr = np.array(f0_estimates, dtype=np.float32)
        p_med = float(np.median(f0_arr))
        p_10 = float(np.percentile(f0_arr, 10))
        p_90 = float(np.percentile(f0_arr, 90))
        semitone_span = 12.0 * float(np.log2(max(p_90, 1.0) / max(p_10, 1.0)))
    else:
        p_med = p_10 = p_90 = semitone_span = 0.0

    if p_med <= 0:
        register = "Unvoiced / Whisper"
    elif p_med < 105:
        register = "Deep Bass (<105 Hz)"
    elif p_med < 145:
        register = "Baritone (105–145 Hz)"
    elif p_med < 185:
        register = "Tenor / Low Alto (145–185 Hz)"
    elif p_med < 235:
        register = "Alto / Mezzo (185–235 Hz)"
    else:
        register = "High / Soprano (235+ Hz)"

    clean_words = [w for w in re.sub(r"<[^>]+>", " ", spoken_text or "").split() if any(c.isalnum() for c in w)]
    wpm = int(round((len(clean_words) / duration_sec) * 60.0)) if clean_words else 0

    return {
        "pitch_hz_median": round(p_med, 1),
        "pitch_hz_p10": round(p_10, 1),
        "pitch_hz_p90": round(p_90, 1),
        "pitch_semitone_span": round(semitone_span, 1),
        "voice_register": register,
        "rms_dbfs": rms_dbfs,
        "words_per_min": wpm,
    }


def apply_studio_pacing(
    audio_np: np.ndarray,
    sr: int = 24000,
    speed: float = 1.0,
    trim_silence: bool = False,
    normalize_audio: bool = False,
) -> np.ndarray:
    """
    Optional post-processor (OFF by default: `speed=1.0, trim_silence=False, normalize_audio=False`
    returns the 100% raw, bit-exact neural waveform from Qwen3-TTS / Maya-1!).
    Only runs silence trimming or ffmpeg `atempo` if explicitly requested.
    """
    if audio_np.size == 0:
        return audio_np

    # Step 1: Trim leading/trailing silence & tighten dead-air pauses ONLY if explicitly enabled
    if trim_silence:
        win = int(sr * 0.02)  # 20ms frame
        if len(audio_np) > win * 6:
            n_frames = len(audio_np) // win
            frames = audio_np[: n_frames * win].reshape(n_frames, win)
            rms = np.sqrt(np.mean(frames ** 2, axis=1))
            thresh = max(0.004, float(np.percentile(rms, 15)) * 1.25)
            is_silent = rms < thresh

            non_silent_idx = np.where(~is_silent)[0]
            if non_silent_idx.size > 0:
                first_f = max(0, int(non_silent_idx[0]) - 2)
                last_f = min(n_frames, int(non_silent_idx[-1]) + 3)
                frames = frames[first_f:last_f]
                is_silent = is_silent[first_f:last_f]
                n_frames = frames.shape[0]

            max_silent_frames = 6  # keep at most 120ms of consecutive internal silence
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

            if keep_chunks:
                audio_np = np.concatenate(keep_chunks, axis=0)

    # Step 2: Pitch-preserving tempo adjustment ONLY if speed != 1.0
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

    if normalize_audio:
        audio_np = normalize_studio_loudness(audio_np, sr=sr)

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

VALID_MAYA1_TAG_SET = {t["tag"] for t in MAYA1_EMOTION_TAGS}

# Common LLM / writer variations mapped to official Maya-1 SNAC tags
MAYA1_TAG_ALIASES = {
    "laughs": "<laugh>",
    "laughing": "<laugh>",
    "chuckles": "<chuckle>",
    "chuckling": "<chuckle>",
    "giggles": "<giggle>",
    "giggling": "<giggle>",
    "sighs": "<sigh>",
    "sighing": "<sigh>",
    "exhales": "<exhale>",
    "gasps": "<gasp>",
    "gasping": "<gasp>",
    "gulps": "<gulp>",
    "snorts": "<snort>",
    "whispers": "<whisper>",
    "whispering": "<whisper>",
    "cries": "<cry>",
    "crying": "<cry>",
    "screams": "<scream>",
    "screaming": "<scream>",
    "sings": "<sing>",
    "singing": "<sing>",
}

# Maps inline tags to natural-language delivery descriptors when routed to Qwen3-TTS
TAG_TO_QWEN_INSTRUCT = {
    "<whisper>": "hushed intimate whisper delivery",
    "<angry>": "fierce angry intense tone",
    "<sarcastic>": "dry sarcastic biting tone",
    "<mischievous>": "playful mischievous tone",
    "<excited>": "high-energy excited enthusiastic tone",
    "<curious>": "inquisitive curious inflections",
    "<disappointed>": "weary disappointed tone",
    "<appalled>": "shocked appalled tone",
    "<cry>": "emotional trembling tearful delivery",
    "<scream>": "urgent raised shouting delivery",
    "<laugh>": "warm amused chuckling delivery",
    "<chuckle>": "warm amused chuckling delivery",
    "<giggle>": "lighthearted giggly delivery",
    "<sigh>": "weary breathy sighing cadence",
}


# Words that already tell Qwen3 VoiceDesign how to emote, so no generic expressiveness cue is added
QWEN_EMOTION_WORDS = (
    "emotion", "expressive", "animated", "dramatic", "intonation", "inflection", "tone", "mood",
    "angry", "furious", "sad", "tearful", "crying", "happy", "cheerful", "joyful", "excited", "nervous",
    "anxious", "scared", "afraid", "panicked", "frantic", "calm", "soothing", "gentle", "sarcastic",
    "playful", "flirty", "seductive", "menacing", "threatening", "weary", "tired", "bored", "deadpan",
    "monotone", "flat", "whisper", "shouting", "yelling", "pleading", "desperate", "confident", "smug",
    "warmly", "coldly", "bitter", "awkward", "embarrassed", "hesitant", "defeated", "baffled", "puzzled",
    "uneasy", "concerned", "worried", "entitled", "irritated", "annoyed", "laughing", "giggly",
)


def sanitize_dialogue_for_model(text: str, model_id: str) -> Tuple[str, List[str]]:
    """
    Formats dialogue text "for the ear, not the eye" based on model family:
      - Normalizes markdown asterisks (`*laughs*` -> `<laugh>` for Maya-1, or removes `*emphasis*` asterisks)
      - For `maya1`: normalizes `<laughs>` / `[laugh]` / `*sighs*` into official `<laugh>` / `<sigh>` SNAC tags
      - For `qwen3_voicedesign` / `qwen3_clone`: extracts any `<tag>` cues so they can enrich `instruct`,
        replaces breath tags (`<sigh>`, `<gasp>`, `<exhale>`) with natural punctuation (`... `),
        and strips all `<...>` brackets so Qwen3 NEVER reads XML tag names out loud!
    Returns `(cleaned_text, extracted_tags)`.
    """
    s = text.strip()
    if not s:
        return "", []

    extracted_tags: List[str] = []

    # Convert *laughs*, *sighs*, etc. into canonical <tag>
    def _replace_asterisk_action(match: re.Match) -> str:
        word = match.group(1).strip().lower()
        if word in MAYA1_TAG_ALIASES:
            return f" {MAYA1_TAG_ALIASES[word]} "
        if f"<{word}>" in VALID_MAYA1_TAG_SET:
            return f" <{word}> "
        # Regular *emphasized* word -> keep word without asterisks
        return match.group(1)

    s = re.sub(r"\*+([^*]+)\*+", _replace_asterisk_action, s)

    # Normalize <laughs>, <sighs>, etc. into canonical Maya-1 tags
    def _normalize_angle_tag(match: re.Match) -> str:
        raw_inner = match.group(1).strip().lower()
        canonical = MAYA1_TAG_ALIASES.get(raw_inner, f"<{raw_inner}>")
        if canonical in VALID_MAYA1_TAG_SET:
            extracted_tags.append(canonical)
            return canonical
        return ""

    s = re.sub(r"<([a-zA-Z_]+)>", _normalize_angle_tag, s)

    if model_id != "maya1":
        # Qwen3-TTS does NOT use inline <tags>; strip them cleanly so it never speaks them aloud!
        for breath_tag in ("<sigh>", "<exhale>", "<gasp>", "<gulp>"):
            s = s.replace(breath_tag, "... ")
        for tag in VALID_MAYA1_TAG_SET:
            s = s.replace(tag, " ")

    # Clean up extra whitespace and awkward punctuation spacing
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.!?])", r"\1", s)
    return s, extracted_tags


def compile_voice_instruct(
    raw_instruct: str,
    model_id: str = "qwen3_voicedesign",
    extracted_tags: Optional[List[str]] = None,
) -> str:
    """
    Model-Specific Prompt Compiler based on Qwen3-TTS & Maya-1 research:
      - Qwen3-TTS VoiceDesign performs best with the 7-Dimension Formula:
        [Identity: Gender/Age/Archetype] + [Pitch & Register] + [Texture & Timbre] + [Pace & Rhythm] + [Emotion & Tone] + [Studio Context]
      - Maya-1 3.3B performs best with its structured `<description="...">` schema:
        "[Role/Character], [Gender] voice in their [Age]s with a [Accent] accent, [pitch] pitch, [timbre] timbre, [pacing] pacing, [emotion] tone at [intensity] intensity."
    Preserves 100% of the user's description while automatically appending any missing acoustic anchors.
    """
    base = (raw_instruct or "").strip().rstrip(".")
    lower = base.lower()
    tags = extracted_tags or []

    # Inject any extracted inline tag emotions when targeting Qwen3
    tag_cues = []
    if model_id != "maya1" and tags:
        for t in tags:
            cue = TAG_TO_QWEN_INSTRUCT.get(t)
            if cue and cue not in tag_cues and cue not in lower:
                tag_cues.append(cue)

    if not base:
        if model_id == "maya1":
            return "Expressive game character, voice in their 30s with a clear accent, medium pitch, warm resonant timbre, brisk conversational pacing, engaging tone at medium intensity."
        return "A clear, expressive game character voice in their 30s, medium resonant pitch, warm articulate vocal texture, natural brisk conversational pacing, engaging studio dialogue delivery."

    has_pitch = any(w in lower for w in ("pitch", "baritone", "bass", "tenor", "alto", "soprano", "deep", "high", "low", "mid-range"))
    has_texture = any(w in lower for w in ("timbre", "texture", "raspy", "gravelly", "smooth", "warm", "crisp", "husky", "breathy", "resonant", "silky", "hoarse", "clear", "smoky"))
    has_pace = any(w in lower for w in ("pace", "pacing", "cadence", "rhythm", "fast", "brisk", "snappy", "steady", "measured", "deliberate", "slow", "rapid", "conversational"))
    has_intensity = "intensity" in lower

    additions: List[str] = []
    if tag_cues:
        additions.extend(tag_cues)

    if model_id == "maya1":
        if not has_pitch:
            additions.append("natural well-defined pitch")
        if not has_texture:
            additions.append("expressive studio vocal timbre")
        if not has_pace:
            additions.append("brisk conversational pacing")
        if not has_intensity:
            additions.append("expressive tone at medium-high intensity")
    else:
        # Qwen3-TTS 7-Dimension completion. No pitch filler: phrases like "well-anchored pitch" ask the
        # model for a steady, monotone contour. When the description names no emotion or delivery style,
        # ask for an emotionally engaged read, since VoiceDesign otherwise defaults to a neutral narrator.
        has_emotion = tag_cues or any(w in lower for w in QWEN_EMOTION_WORDS)
        if not has_texture:
            additions.append("articulate vocal timbre")
        if not has_pace:
            additions.append("natural conversational pacing")
        if not has_emotion:
            additions.append("emotionally engaged, expressive delivery with lively, varied intonation")
        if "studio" not in lower and "dialogue" not in lower and "mic" not in lower:
            additions.append("clean close-mic recording")

    if additions:
        return f"{base}, {', '.join(additions)}."
    return f"{base}."


def optimize_character_prompts(
    concept: str,
    name: str = "Character",
    gender_age_hint: str = "",
) -> Dict[str, str]:
    """
    Takes a short or rough character concept (e.g. 'grumpy dwarven blacksmith' or 'sarcastic cyberpunk hacker')
    and returns ready-to-use, research-optimized prompts for both Qwen3-TTS VoiceDesign and Maya-1,
    plus a phonetically balanced 10-second Golden Reference Sample for zero-drift Qwen3 Base Voice Cloning.
    """
    c = concept.strip().rstrip(".")
    low = c.lower()

    # Infer helpful defaults if user gave only a 2-3 word concept
    gender = "female" if any(w in low for w in ("female", "woman", "girl", "queen", "witch", "sorceress", "sister", "lady", "mother", "priestess", "heroine")) else (
        "male" if any(w in low for w in ("male", "man", "boy", "king", "wizard", "dwarf", "blacksmith", "knight", "father", "lord", "brother")) else "distinctive"
    )
    age = "in their 50s" if any(w in low for w in ("old", "elder", "ancient", "veteran", "weathered", "grumpy", "blacksmith", "mentor")) else (
        "in their 20s" if any(w in low for w in ("young", "teen", "bubbly", "rookie", "apprentice", "energetic", "rogue")) else "in their 30s"
    )
    accent = "Scottish accent" if "dwarf" in low or "scottish" in low else (
        "British accent" if any(w in low for w in ("british", "villain", "noble", "aristocrat", "wizard", "rogue", "knight")) else "neutral American accent"
    )
    pitch = "deep resonant low pitch" if any(w in low for w in ("dwarf", "villain", "warrior", "giant", "commander", "noir", "baritone", "deep")) else (
        "bright mid-high pitch" if any(w in low for w in ("bubbly", "energetic", "alchemist", "companion", "young")) else "confident mid-range pitch"
    )
    texture = "gravelly raspy weathered vocal texture" if any(w in low for w in ("dwarf", "grumpy", "rough", "veteran", "blacksmith", "warrior", "pirate")) else (
        "smooth, sultry, slightly husky studio timbre" if any(w in low for w in ("sexy", "sultry", "alluring", "smooth", "charismatic", "flirty")) else "crisp, expressive, articulate vocal texture"
    )

    qwen_prompt = compile_voice_instruct(
        f"{c}. A {gender} voice {age} with a {accent}, {pitch}, {texture}, brisk natural conversational pacing, expressive character acting for studio game dialogue",
        model_id="qwen3_voicedesign",
    )
    maya_prompt = compile_voice_instruct(
        f"{c}, {gender.capitalize()} voice {age} with a {accent}, {pitch}, {texture}, brisk conversational pacing, expressive character tone at medium-high intensity",
        model_id="maya1",
    )

    # 10-second phonetically rich Golden Reference sentence (ideal for Qwen3-TTS Base ICL Voice Cloning)
    golden_sample_qwen = (
        f"Listen closely, friend. Everyone in this place has a secret they are trying to hide! "
        f"And if we want to make it out before sunrise... well, you had better stick right beside me, okay?"
    )
    golden_sample_maya = (
        f"Listen closely, friend. <sigh> Everyone in this place has a secret they are trying to hide, "
        f"<chuckle> and if we want to make it out before sunrise, you had better stick right beside me."
    )

    return {
        "name": name,
        "raw_concept": concept,
        "instruct_qwen": qwen_prompt,
        "instruct_maya": maya_prompt,
        "golden_sample_qwen": golden_sample_qwen,
        "golden_sample_maya": golden_sample_maya,
    }



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

    @staticmethod
    def pack_snac_to_7(levels: List[List[int]]) -> List[int]:
        """
        Inverse of `unpack_snac_from_7`: packs 3 hierarchical SNAC code levels `[l1, l2, l3]`
        into Maya-1's 7-token-per-frame interleaved format for SNAC Audio-Prefix In-Context Voice Locking!
        """
        l1, l2, l3 = levels[0], levels[1], levels[2]
        frames = min(len(l1), len(l2) // 2, len(l3) // 4)
        packed: List[int] = []
        for i in range(frames):
            s0 = CODE_TOKEN_OFFSET + 0 * 4096 + int(l1[i])
            s1 = CODE_TOKEN_OFFSET + 1 * 4096 + int(l2[2 * i])
            s2 = CODE_TOKEN_OFFSET + 2 * 4096 + int(l3[4 * i])
            s3 = CODE_TOKEN_OFFSET + 3 * 4096 + int(l3[4 * i + 1])
            s4 = CODE_TOKEN_OFFSET + 4 * 4096 + int(l2[2 * i + 1])
            s5 = CODE_TOKEN_OFFSET + 5 * 4096 + int(l3[4 * i + 2])
            s6 = CODE_TOKEN_OFFSET + 6 * 4096 + int(l3[4 * i + 3])
            packed.extend([s0, s1, s2, s3, s4, s5, s6])
        return packed

    def encode_reference_audio_tokens(self, ref_audio_source: Any, max_ref_sec: float = 5.5) -> List[int]:
        """
        Encodes a saved Character Voice `.wav` (or numpy array) into Maya-1 7-slot SNAC tokens so Maya-1 can
        continue the exact speaker's voice (and previous line's context) while executing `<laugh>`, `<sigh>`, `<gasp>` tags!
        """
        if isinstance(ref_audio_source, np.ndarray):
            audio_np = ref_audio_source.astype(np.float32)
            sr = self.sample_rate
        else:
            audio_np, sr = sf.read(str(ref_audio_source), dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != self.sample_rate and audio_np.size > 0:
            # Simple linear interpolation to 24kHz if needed
            duration = audio_np.shape[0] / sr
            target_len = int(duration * self.sample_rate)
            audio_np = np.interp(
                np.linspace(0.0, 1.0, target_len, endpoint=False),
                np.linspace(0.0, 1.0, audio_np.shape[0], endpoint=False),
                audio_np,
            ).astype(np.float32)
        max_samples = int(max_ref_sec * self.sample_rate)
        if audio_np.shape[0] > max_samples:
            # Keep the most recent `max_ref_sec` seconds so any chained previous line stays intact!
            audio_np = audio_np[-max_samples:]
        audio_mx = mx.array(audio_np, dtype=mx.float32)[None, None, :]
        codes = self.snac.encode(audio_mx)
        mx.eval(codes)
        levels = [np.array(c.squeeze(0), dtype=np.int32).tolist() for c in codes]
        return self.pack_snac_to_7(levels)

    def generate(
        self,
        text: str,
        instruct: str,
        temperature: float = 0.45,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        max_tokens: int = 2048,
        ref_audio_path: Optional[Any] = None,
        ref_text: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = time.perf_counter()
        ref_snac_tokens: List[int] = []
        ref_frames: int = 0

        has_valid_ref = (
            ref_audio_path is not None
            and ref_text
            and (isinstance(ref_audio_path, np.ndarray) or Path(str(ref_audio_path)).exists())
        )
        if has_valid_ref:
            ref_dur = (
                ref_audio_path.shape[0] / self.sample_rate
                if isinstance(ref_audio_path, np.ndarray)
                else sf.info(str(ref_audio_path)).duration
            )
            if ref_dur <= 6.5:
                raw_snac = self.encode_reference_audio_tokens(ref_audio_path, max_ref_sec=6.5)
                # Trim trailing 2 SNAC frames (silence tail) so Maya-1 doesn't immediately emit CODE_END_TOKEN_ID!
                if len(raw_snac) > 28:
                    raw_snac = raw_snac[:-14]
                ref_snac_tokens = raw_snac
                ref_frames = len(ref_snac_tokens) // SNAC_TOKENS_PER_FRAME
                combined_text = f"{ref_text.strip().rstrip('.!?')}, {text.strip()}"
                prompt_str = self.build_prompt(instruct, combined_text)
            else:
                prompt_str = self.build_prompt(instruct, text)
        else:
            prompt_str = self.build_prompt(instruct, text)

        encoded = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        if ref_snac_tokens:
            encoded = encoded + ref_snac_tokens
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
            if tok == CODE_END_TOKEN_ID:
                if len(generated_ids) >= 28:
                    break
                # Ignore premature EOS before 4 SNAC frames (28 tokens)
                continue
            generated_ids.append(tok)
            if len(generated_ids) % 100 == 0:
                mx.clear_cache()

        snac_tokens = self.extract_snac_codes(generated_ids)
        if ref_snac_tokens:
            # Decode together with reference prefix for seamless SNAC convolutional state, then slice!
            full_levels = self.unpack_snac_from_7(ref_snac_tokens + snac_tokens)
            gen_levels = self.unpack_snac_from_7(snac_tokens)
            if not gen_levels[0]:
                raise RuntimeError("Maya-1 did not produce enough acoustic tokens.")
            codes_mx = [mx.array(level, dtype=mx.int32)[None, :] for level in full_levels]
            audio_hat = self.snac.decode(codes_mx).squeeze()
            mx.eval(audio_hat)
            full_audio_np = np.array(audio_hat, dtype=np.float32)
            # Each SNAC frame (1 l1 token) corresponds to 2048 audio samples at 24kHz
            slice_sample = ref_frames * 2048
            audio_np = full_audio_np[slice_sample:] if full_audio_np.shape[0] > slice_sample else full_audio_np
            levels = gen_levels
        else:
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
            "snac_voice_locked": bool(ref_snac_tokens),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        }


RVQ_LOCK_LIMITS = {"temperature": 0.65, "top_p": 0.90, "top_k": 40}


def _install_qwen3_timbre_lock(model: Any) -> None:
    """
    Separates Qwen3-TTS's Prosody/Cadence sampling (`code_0`, where `suppress_tokens` is provided)
    from its 15 Residual Vector Quantization (RVQ) Acoustic Timbre Codebooks (`code_1..code_15`,
    where `suppress_tokens is None` inside `code_predictor`).
    By clamping the 15 acoustic timbre codebooks to lower-entropy sampling (`RVQ_LOCK_LIMITS`), the character's vocal cords, formants, and accent stay 100%
    locked to the Golden Reference across every line, while `code_0` remains expressive!
    """
    talker = getattr(model, "talker", None)
    if talker is None or getattr(talker, "_timbre_lock_installed", False):
        return
    orig_sample_token = getattr(model, "_sample_token", None)
    if orig_sample_token is None:
        return

    def _timbre_locked_sample_token(logits, *args, **kwargs):
        suppress_tokens = kwargs.get("suppress_tokens", None)
        if suppress_tokens is None and getattr(model, "_lock_rvq_timbre", False):
            # Sampling RVQ acoustic timbre codebooks code_1..code_15: clamp entropy!
            # Near-greedy residual codebooks (the old 0.30 / 0.75 / 20 clamp) strip breath and grit and
            # give a smoothed, "processed" texture; this milder clamp still keeps timbre stable.
            if "temperature" in kwargs:
                kwargs["temperature"] = min(float(kwargs["temperature"]), RVQ_LOCK_LIMITS["temperature"])
            if "top_p" in kwargs:
                kwargs["top_p"] = min(float(kwargs["top_p"]), RVQ_LOCK_LIMITS["top_p"])
            if "top_k" in kwargs:
                kwargs["top_k"] = min(int(kwargs["top_k"]), RVQ_LOCK_LIMITS["top_k"])
        return orig_sample_token(logits, *args, **kwargs)

    model._sample_token = _timbre_locked_sample_token
    model._lock_rvq_timbre = True
    talker._timbre_lock_installed = True


def _build_rolling_context_reference(
    golden_audio_path: Optional[str],
    golden_text: Optional[str],
    prev_audio_path: Optional[str],
    prev_text: Optional[str],
    sample_rate: int = 24000,
    max_prev_sec: float = 8.0,
    max_total_sec: float = 18.0,
) -> Tuple[Optional[mx.array], Optional[str], Optional[np.ndarray]]:
    """
    Builds a Hybrid Context-Aware Reference `(ref_audio_mx, combined_ref_text, ref_audio_np)` that combines:
      1. The character's permanent Golden Reference Anchor (so vocal cords/timbre never drift over 20+ lines)
      2. A 140ms natural micro-pause
      3. The immediately preceding dialogue line's audio + transcript (`prev_audio_path` + `prev_text`)
    When fed into `Qwen3-TTS Base` (or `Maya-1` SNAC prefix), the transformer literally hears AND reads
    what the character just said in Line N-1 before speaking Line N, creating seamless conversational
    prosody, pitch continuity, and emotional momentum!
    """
    if not golden_audio_path or not golden_text or not Path(golden_audio_path).exists():
        return None, None, None

    g_wav, g_sr = sf.read(golden_audio_path, dtype="float32")
    if g_wav.ndim > 1:
        g_wav = g_wav.mean(axis=1)
    g_text = golden_text.strip()

    if not prev_audio_path or not prev_text or not Path(prev_audio_path).exists():
        return mx.array(g_wav, dtype=mx.float32), g_text, g_wav

    p_wav, p_sr = sf.read(prev_audio_path, dtype="float32")
    if p_wav.ndim > 1:
        p_wav = p_wav.mean(axis=1)
    p_text = prev_text.strip()

    # ICL cloning needs ref audio and ref text to line up exactly. Cutting the audio while keeping the full
    # transcript makes the model never find the end of the reference and spiral into runaway generation,
    # so a previous line that is too long is skipped (golden anchor only) rather than truncated.
    # The old 5s limit skipped chaining for most emotional lines (they run 5-8s), so the momentum
    # from the previous line was usually lost.
    if p_wav.shape[0] > int(max_prev_sec * sample_rate) or g_wav.shape[0] + p_wav.shape[0] > int(max_total_sec * sample_rate):
        return mx.array(g_wav, dtype=mx.float32), g_text, g_wav

    gap = np.zeros(int(0.14 * sample_rate), dtype=np.float32)
    combined_np = np.concatenate([g_wav, gap, p_wav], axis=0).astype(np.float32)
    combined_text = f"{g_text} {p_text}".strip()
    return mx.array(combined_np, dtype=mx.float32), combined_text, combined_np


# Sampling defaults. code_0 (prosody) needs enough entropy to move pitch and timing; at the old clone
# defaults (0.50 / 0.82) it settled into a narrow, read-aloud contour. 1.5 matches the repetition penalty
# mlx-audio used to force on ICL, which keeps long chained references from looping.
QWEN_DESIGN_DEFAULTS = {"temperature": 0.80, "top_p": 0.95, "repetition_penalty": 1.05}
QWEN_CLONE_DEFAULTS = {"temperature": 0.75, "top_p": 0.92, "repetition_penalty": 1.5}


def _split_for_synthesis(text: str, max_chars: int = 220) -> List[str]:
    """
    Splits long text into sentence groups of at most ~`max_chars`. One ICL pass over a long paragraph
    tends to rush and flatten out (and is the most likely to spiral), so each group is generated on its own.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    sentences = [x.strip() for x in re.split(r"(?<=[.!?…])\s+", text) if x.strip()]
    groups: List[str] = []
    cur = ""
    for sent in sentences:
        if cur and len(cur) + 1 + len(sent) > max_chars:
            groups.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        groups.append(cur)
    return groups


def _estimate_max_tokens(text: str, tokens_per_sec: float = 12.5) -> int:
    """
    Estimates a generous upper bound on codec tokens for `text` (~2.3 words/sec of speech, doubled, plus 2s margin)
    so a spiralling generation is killed early instead of running to the model's default limit.
    """
    words = max(1, len(text.split()))
    expected_sec = words / 2.3
    return int((expected_sec * 2.0 + 2.0) * tokens_per_sec)


def _derive_deterministic_seed(key_str: str) -> int:
    """Derives a stable 31-bit integer seed from a character's reference audio path or name."""
    h = 2166136261
    for ch in (key_str or "voiceforge"):
        h = ((h ^ ord(ch)) * 16777619) & 0x7FFFFFFF
    return max(1, h)


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
                    _install_qwen3_timbre_lock(self.active_model)

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
        trim_silence: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        prev_audio_path: Optional[str] = None,
        prev_text: Optional[str] = None,
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        # Ensure requested model is active
        if self.active_model_id != model_id or self.active_model is None:
            self.load_model(model_id)

        with self._lock:
            info = MODEL_REGISTRY[model_id]
            t0 = time.perf_counter()

            # Clean spoken text ("for the ear, not the eye") and compile model-specific instruct prompt
            clean_text, extracted_tags = sanitize_dialogue_for_model(text, model_id=model_id)
            compiled_instruct = compile_voice_instruct(instruct, model_id=model_id, extracted_tags=extracted_tags)

            # Set deterministic MLX random seed if provided or if locking to a reference character
            effective_seed = seed
            if effective_seed is None and ref_audio_path:
                effective_seed = _derive_deterministic_seed(str(ref_audio_path))
            if effective_seed is not None:
                mx.random.seed(int(effective_seed))

            # Build Hybrid Context-Aware Reference (Golden Anchor + Previous Line Audio/Text) when available!
            clean_ref_text = None
            if ref_text:
                clean_ref_text, _ = sanitize_dialogue_for_model(ref_text, model_id="qwen3_clone")
            clean_prev_text = None
            if prev_text:
                clean_prev_text, _ = sanitize_dialogue_for_model(prev_text, model_id="qwen3_clone")

            ref_audio_mx, combined_ref_text, ref_audio_np = _build_rolling_context_reference(
                golden_audio_path=ref_audio_path,
                golden_text=clean_ref_text,
                prev_audio_path=prev_audio_path,
                prev_text=clean_prev_text,
                sample_rate=24000,
            )
            used_prev_context = bool(
                prev_audio_path and clean_prev_text and ref_audio_mx is not None
                and combined_ref_text and combined_ref_text != clean_ref_text
            )

            if info["family"] == "maya1":
                temp = temperature if temperature is not None else 0.42
                tp = top_p if top_p is not None else 0.88
                rp = repetition_penalty if repetition_penalty is not None else 1.15
                # For Maya-1 SNAC prefix, use the immediately preceding line if available (3-5s),
                # or the Golden Reference if <= 6.5s, so `ref_text` and `ref_audio` stay 100% aligned!
                maya_ref_src = prev_audio_path if (prev_audio_path and clean_prev_text) else ref_audio_path
                maya_ref_txt = clean_prev_text if (prev_audio_path and clean_prev_text) else clean_ref_text
                audio_np, metrics = self.active_model.generate(
                    text=clean_text,
                    instruct=compiled_instruct,
                    temperature=temp,
                    top_p=tp,
                    repetition_penalty=rp,
                    ref_audio_path=maya_ref_src,
                    ref_text=maya_ref_txt,
                )
                audio_np = apply_studio_pacing(audio_np, 24000, speed=speed, trim_silence=trim_silence)
                acoustics = analyze_voice_acoustics(audio_np, 24000, clean_text)
                metrics.update(acoustics)
                metrics["duration_sec"] = round(audio_np.shape[0] / 24000, 2)
                metrics["speed"] = round(speed, 2)
                metrics["temperature"] = round(temp, 2)
                metrics["top_p"] = round(tp, 2)
                metrics["seed"] = effective_seed
                metrics["context_chained"] = used_prev_context
                metrics["compiled_instruct"] = compiled_instruct
                return audio_np, 24000, metrics

            # Qwen3-TTS family
            token_cap = _estimate_max_tokens(clean_text)
            base_seed = effective_seed if effective_seed is not None else int(time.time() * 1000) & 0x7FFFFFFF

            def _guarded(make_gen, attempt_specs):
                """Run attempts in order until one finishes under the token cap (a runaway never hits EOS)."""
                for attempt_idx, spec in enumerate(attempt_specs):
                    mx.random.seed(int(spec["seed"]))
                    out = list(make_gen(spec))
                    if not any(r.token_count >= token_cap for r in out):
                        return out, attempt_idx
                    print(f"[voiceforge] runaway generation killed at {token_cap} tokens (attempt {attempt_idx + 1}/{len(attempt_specs)}): {clean_text[:50]!r}", flush=True)
                raise RuntimeError(f"Generation spiralled past the {token_cap}-token limit on every attempt: {clean_text[:60]!r}")

            if info["mode"] == "voice_design":
                temp = temperature if temperature is not None else QWEN_DESIGN_DEFAULTS["temperature"]
                tp = top_p if top_p is not None else QWEN_DESIGN_DEFAULTS["top_p"]
                rp = repetition_penalty if repetition_penalty is not None else QWEN_DESIGN_DEFAULTS["repetition_penalty"]
                self.active_model._lock_rvq_timbre = False
                if not instruct.strip():
                    raise ValueError("Please provide a character/voice description for Qwen3 VoiceDesign.")
                # VoiceDesign had no length cap at all, so a runaway ran to 2048 tokens (~160s of babble)
                # and could get locked into the roster as a character's golden reference.
                results, _ = _guarded(
                    lambda spec: self.active_model.generate_voice_design(
                        text=clean_text,
                        instruct=compiled_instruct,
                        language=language,
                        temperature=temp,
                        top_p=tp,
                        repetition_penalty=rp,
                        max_tokens=token_cap,
                    ),
                    [{"seed": base_seed}, {"seed": base_seed + 1}, {"seed": base_seed + 2}],
                )
            elif info["mode"] == "voice_clone":
                temp = temperature if temperature is not None else QWEN_CLONE_DEFAULTS["temperature"]
                tp = top_p if top_p is not None else QWEN_CLONE_DEFAULTS["top_p"]
                rp = repetition_penalty if repetition_penalty is not None else QWEN_CLONE_DEFAULTS["repetition_penalty"]
                self.active_model._lock_rvq_timbre = True
                if not ref_audio_path or not ref_text:
                    raise ValueError("Voice Lock / Clone mode requires selecting a saved character voice (reference audio + transcript).")
                effective_ref_audio = ref_audio_mx if ref_audio_mx is not None else ref_audio_path
                effective_ref_text = combined_ref_text if combined_ref_text else clean_ref_text
                golden_audio = load_audio(ref_audio_path, sample_rate=self.active_model.sample_rate)

                def _icl(spec, chunk_text):
                    # Calls the ICL generator directly: model.generate() silently raises the repetition
                    # penalty to >= 1.5 for ICL, so the Rep. Penalty slider had no effect in clone mode.
                    return self.active_model._generate_icl(
                        text=chunk_text,
                        ref_audio=load_audio(spec["audio"], sample_rate=self.active_model.sample_rate),
                        ref_text=spec["text"],
                        language=language,
                        temperature=temp,
                        top_p=tp,
                        repetition_penalty=rp,
                        max_tokens=token_cap,
                    )

                chunks_text = _split_for_synthesis(clean_text)
                results = []
                for chunk_idx, chunk_text in enumerate(chunks_text):
                    token_cap = _estimate_max_tokens(chunk_text)
                    # First chunk uses the chained context; later chunks re-anchor on the golden reference
                    # alone so a long line can't drift away from the character's voice.
                    if chunk_idx == 0:
                        specs = [
                            {"audio": effective_ref_audio, "text": effective_ref_text, "seed": base_seed},
                            {"audio": effective_ref_audio, "text": effective_ref_text, "seed": base_seed + 1},
                            {"audio": golden_audio, "text": clean_ref_text, "seed": base_seed + 2},
                        ]
                    else:
                        specs = [
                            {"audio": golden_audio, "text": clean_ref_text, "seed": base_seed + 10 * chunk_idx},
                            {"audio": golden_audio, "text": clean_ref_text, "seed": base_seed + 10 * chunk_idx + 1},
                        ]
                    chunk_results, attempt_idx = _guarded(lambda spec: _icl(spec, chunk_text), specs)
                    if chunk_idx == 0 and attempt_idx == 2:
                        used_prev_context = False
                    results.extend(chunk_results)
            else:
                raise ValueError(f"Unsupported mode: {info['mode']}")

            if not results:
                raise RuntimeError("Model returned no audio segments.")

            chunks = [np.array(r.audio, dtype=np.float32) for r in results if r.audio is not None]
            if len(chunks) > 1:
                gap = np.zeros(int(0.18 * self.active_model.sample_rate), dtype=np.float32)
                joined = [chunks[0]]
                for c in chunks[1:]:
                    joined.extend([gap, c])
                audio_np = np.concatenate(joined, axis=0)
            else:
                audio_np = chunks[0]
            sr = self.active_model.sample_rate
            audio_np = apply_studio_pacing(audio_np, sr, speed=speed, trim_silence=trim_silence)
            acoustics = analyze_voice_acoustics(audio_np, sr, clean_text)
            elapsed = time.perf_counter() - t0
            duration = audio_np.shape[0] / sr
            total_tokens = sum(r.token_count for r in results)

            out_metrics = {
                "duration_sec": round(duration, 2),
                "elapsed_sec": round(elapsed, 2),
                "rtf": round(duration / elapsed, 2) if elapsed > 0 else 0,
                "tokens": total_tokens,
                "speed": round(speed, 2),
                "temperature": round(temp, 2),
                "top_p": round(tp, 2),
                "seed": base_seed,
                "chunks": len(results),
                "rvq_timbre_locked": bool(getattr(self.active_model, "_lock_rvq_timbre", False)),
                "context_chained": used_prev_context,
                "compiled_instruct": compiled_instruct,
                "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
            }
            out_metrics.update(acoustics)
            return audio_np, sr, out_metrics
