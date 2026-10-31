"""
FastAPI backend for VoiceForge Studio.
Provides clean REST & OpenAI-compatible endpoints so both humans (via the Web UI)
and external LLMs / AI agents / game engines can control every feature programmatically:
  - Model hot-swapping (Qwen3-TTS VoiceDesign <-> Maya-1 3.3B <-> Qwen3 Base Voice Lock)
  - Designing new voices from text descriptions
  - Expressive acting with Maya-1 inline emotion tags
  - Locking takes into reusable NPC Character Voices
  - Generating zero-drift dialogue from locked characters
  - Direct WAV streaming & OpenAI-compatible `/v1/audio/speech`
  - Auto-generated OpenAPI / Swagger docs at `/docs` and `/openapi.json`
"""

import io
import json
import os
import random
import re
import shutil
import subprocess
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import (
    VoiceForgeEngine,
    MODEL_REGISTRY,
    MAYA1_EMOTION_TAGS,
    analyze_voice_acoustics,
    compile_voice_instruct,
    sanitize_dialogue_for_model,
    optimize_character_prompts,
    _emotion_temperature_boost,
)

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
TAKES_DIR = DATA_DIR / "takes"
CHARACTERS_DIR = DATA_DIR / "characters"
SCENES_DIR = DATA_DIR / "scenes"
STATIC_DIR = BASE_DIR / "static"

for d in (DATA_DIR, TAKES_DIR, CHARACTERS_DIR, SCENES_DIR, STATIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

TAKES_JSON = DATA_DIR / "takes.json"
CHARACTERS_JSON = DATA_DIR / "characters.json"
SCENES_JSON = DATA_DIR / "scenes.json"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


CHARACTER_PRESETS = [
    {
        "id": "preset_dwarf_blacksmith",
        "name": "Torvin the Ironwright (Dwarven Blacksmith)",
        "category": "Fantasy NPC",
        "instruct_qwen": "An elderly, weathered male dwarven blacksmith in his 60s with a deep, gravelly baritone voice, rough raspy texture, natural conversational pacing, and a cynical Scottish-tinged accent.",
        "instruct_maya": "Rough fantasy dwarf character, Male voice in their 60s with a Scottish accent, very low pitch, gravelly raspy timbre, natural conversational pacing, grumpy tone at medium intensity.",
        "sample_text_qwen": "Keep your coin, traveler. That blade was forged in the old fires, and nothing in these mountains is for sale anymore.",
        "sample_text_maya": "Keep your coin, traveler <snort> that blade was forged in the old fires <sigh> and nothing in these mountains is for sale anymore.",
    },
    {
        "id": "preset_cyber_rogue",
        "name": "Nyx (Cyberpunk Netrunner / Rogue)",
        "category": "Sci-Fi / Action",
        "instruct_qwen": "A sharp, witty 28-year-old female rogue with a sleek British accent, medium-low husky pitch, confident and sarcastic tone, brisk conversational pacing.",
        "instruct_maya": "Sarcastic female rogue character, Female voice in her late 20s with a British accent, slightly husky timbre, brisk pacing, sarcastic and mischievous tone.",
        "sample_text_qwen": "Oh, brilliant plan! Let's just walk right through the main security grid. I'm sure the automated turrets won't mind at all.",
        "sample_text_maya": "Oh, brilliant plan! <laugh> Let's just walk right through the main security grid <sigh> I'm sure the automated turrets won't mind at all.",
    },
    {
        "id": "preset_dark_villain",
        "name": "Malakor (Ancient Lich / Main Villain)",
        "category": "Boss / Antagonist",
        "instruct_qwen": "A sinister, ancient male villain in his 50s with a resonant, aristocratic British accent, cold menacing whispery baritone, dramatic cadence, and dark authority.",
        "instruct_maya": "Dark villain character, Male voice in their 40s with a British accent. low pitch, gravelly timbre, dramatic pacing, angry menacing tone at high intensity.",
        "sample_text_qwen": "You dare step into my sanctum, mortal? Every hero before you has knelt in ashes, and you shall be no different.",
        "sample_text_maya": "You dare step into my sanctum, mortal? <chuckle> How amusing. <whisper> Every hero before you has knelt in ashes.",
    },
    {
        "id": "preset_energetic_companion",
        "name": "Pip (Excitable Alchemist Companion)",
        "category": "Companion",
        "instruct_qwen": "A cheerful, energetic young female alchemist in her early 20s with a bright, clear American voice, slightly high pitch, fast excited pacing, and bubbly enthusiasm.",
        "instruct_maya": "Energetic young female companion in her 20s with an American accent, bright clear timbre, fast excited pacing, curious and cheerful tone.",
        "sample_text_qwen": "Wait, wait, don't touch that vial! It's either going to heal all your wounds or turn your boots into frogs!",
        "sample_text_maya": "<gasp> Wait, wait, don't touch that vial! <giggle> It's either going to heal all your wounds or turn your boots into frogs!",
    },
    {
        "id": "preset_noir_detective",
        "name": "Detective Cole (Gritty Noir Narrator)",
        "category": "Narrator / Noir",
        "instruct_qwen": "A tired 45-year-old male noir detective, late-night smoky baritone voice, neutral American accent, hushed studio microphone presence, steady conversational pacing.",
        "instruct_maya": "Tired male detective in his 40s with an American accent, low smoky baritone, steady conversational pacing, calm exhausted delivery.",
        "sample_text_qwen": "Rain was hammering the neon glass outside my office when she walked in. In this city, a knock after midnight never brings good news.",
        "sample_text_maya": "<exhale> Rain was hammering the neon glass outside my office when she walked in. <sigh> In this city, a knock after midnight never brings good news.",
    },
]

app = FastAPI(
    title="VoiceForge Studio API — Apple Silicon Local TTS (Qwen3-TTS & Maya-1)",
    description=(
        "Local 24kHz Text-to-Speech, Voice Design, Expressive Acting, and Zero-Shot Character Cloning API "
        "powered by Apple Metal MLX. Designed for both human studio use and autonomous LLM / game-engine tool calling."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = VoiceForgeEngine()


# ============================================================================
# Pydantic Schemas (Documented for LLM Tool Use & OpenAPI)
# ============================================================================

class LoadModelRequest(BaseModel):
    model_id: str = Field(
        ...,
        description="Target engine ID to load into Metal VRAM: 'qwen3_voicedesign', 'maya1', or 'qwen3_clone'.",
        examples=["qwen3_voicedesign", "maya1", "qwen3_clone"],
    )


class SynthesizeRequest(BaseModel):
    model_id: str = Field(
        "qwen3_voicedesign",
        description="Engine ID: 'qwen3_voicedesign' (describe voice), 'maya1' (describe voice + inline <laugh>/<sigh> tags), or 'qwen3_clone' (use locked character_id).",
    )
    text: str = Field(
        ...,
        description="The dialogue line to speak. When using 'maya1', you can embed inline tags like <laugh>, <sigh>, <whisper>, <gasp>, <angry>.",
    )
    instruct: str = Field(
        "",
        description="Natural language description of the person and their voice (age, gender, accent, pitch, vocal texture, pacing). Required for 'qwen3_voicedesign' and 'maya1'.",
    )
    language: str = Field(
        "english",
        description="Spoken language ('english', 'auto', 'chinese', 'japanese', 'korean', 'french', 'german', 'spanish').",
    )
    speed: float = Field(
        1.0,
        ge=0.5,
        le=2.0,
        description="Optional speaking speed multiplier (1.0 = pure unprocessed neural output by default; set >1.0 only if tempo stretch is desired).",
    )
    trim_silence: bool = Field(
        False,
        description="Optional silence trimming (False = pure unprocessed neural output by default).",
    )
    temperature: Optional[float] = Field(
        None,
        description="Native model sampling temperature (defaults to 0.95 for Qwen3 Clone, 0.85 for Qwen3 Design, plus a dynamic boost on emotional lines; 0.42 for Maya-1).",
    )
    top_p: Optional[float] = Field(
        None,
        description="Native nucleus sampling top_p (defaults to 0.82 for Qwen3 Clone, 0.95 for Qwen3 Design, 0.88 for Maya-1).",
    )
    repetition_penalty: Optional[float] = Field(
        None,
        description="Native repetition penalty (defaults to 1.5 for Qwen3 Clone, 1.05 for Qwen3 Design, 1.15 for Maya-1).",
    )
    seed: Optional[int] = Field(
        None,
        description="Optional deterministic MLX random seed. In `qwen3_clone`, automatically defaults to a stable per-character seed if omitted.",
    )
    dynamic_temperature: bool = Field(
        True,
        description="Qwen3 only: raise temperature by up to +0.15 on emotionally charged lines (exclamations, ellipses, shouting, emotion tags). The effective value is reported in metrics.",
    )
    character_id: Optional[str] = Field(
        None,
        description="ID or exact name of a locked character from `/api/characters` when using `model_id='qwen3_clone'` (or for Maya-1 SNAC reference locking).",
    )
    prev_take_id: Optional[str] = Field(
        None,
        description="Optional ID of the preceding dialogue take (`take_id`) to chain into the ICL reference context so this line knows how the previous line sounded and what it said!",
    )
    prev_audio_path: Optional[str] = Field(
        None,
        description="Optional path to preceding line's WAV file for context chaining.",
    )
    prev_text: Optional[str] = Field(
        None,
        description="Optional transcript of the preceding line for context chaining.",
    )
    blend_with: Optional[str] = Field(
        None,
        description="Emotion blend: ID or name of a second voice of the same person (e.g. 'Tamair (Sad)'). Its clip is joined onto this voice's reference so the read lands between the two emotions. Qwen3 clone only.",
    )
    blend_strength: str = Field(
        "subtle",
        description="'subtle' keeps this voice's emotion in front (blend clip heard first); 'strong' lets the blend clip lead (heard last, which pulls hardest).",
    )


class DesignAndLockRequest(BaseModel):
    """Single-call LLM workflow: invent a voice from description, synthesize a golden sample, and lock it as a reusable character."""
    name: str = Field(..., description="Name of the new game character (e.g., 'Captain Vane').")
    instruct: str = Field(..., description="Detailed voice description (age, gender, accent, timbre, pitch, pacing).")
    sample_text: str = Field(
        "Keep your eyes sharp. Nothing in this sector is what it seems, and we only get one shot at this.",
        description="Reference sentence the character will speak to establish their vocal timbre (5-12 seconds recommended).",
    )
    design_model_id: str = Field(
        "qwen3_voicedesign",
        description="Which voice-design engine to generate the golden sample with ('qwen3_voicedesign' or 'maya1').",
    )
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speaking speed multiplier (1.0 = pure unprocessed Golden Reference).")
    trim_silence: bool = Field(False, description="Keep False so the Golden Reference waveform is 100% pure for ICL tokenization.")
    temperature: Optional[float] = Field(None, description="Optional native sampling temperature for designing the voice.")
    top_p: Optional[float] = Field(None, description="Optional native top_p for designing the voice.")
    seed: Optional[int] = Field(None, description="Optional deterministic seed for designing the voice.")


class SaveCharacterFromTakeRequest(BaseModel):
    take_id: str = Field(..., description="ID of an existing generated take from `/api/takes`.")
    name: str = Field(..., description="Character name to save in the Character Voice Locker.")
    description: str = Field("", description="Optional notes or voice description.")


class CharacterFromRecordingRequest(BaseModel):
    """Register an external voice audio recording (.wav, .m4a, .mp3, .ogg, .flac) as a locked character."""
    name: str = Field(..., description="Character name to register in the Character Voice Locker.")
    audio_path: str = Field(..., description="Local path to audio recording.")
    ref_text: Optional[str] = Field(None, description="Exact spoken transcript. If omitted, automatically transcribed via on-device Whisper.")
    description: Optional[str] = Field(None, description="Optional notes or character description.")
    start_sec: Optional[float] = Field(None, description="Optional start timestamp in seconds for reference clip extraction.")
    end_sec: Optional[float] = Field(None, description="Optional end timestamp in seconds for reference clip extraction.")


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible `/v1/audio/speech` request schema for drop-in LLM / agent integration."""
    model: str = Field(
        "qwen3_voicedesign",
        description="Model ID ('qwen3_voicedesign', 'maya1', 'qwen3_clone', or 'tts-1').",
    )
    input: str = Field(..., description="The text to generate audio for.")
    voice: str = Field(
        "default",
        description="Either a locked `character_id` / character name (automatically uses `qwen3_clone`), a preset ID, or a natural-language voice description!",
    )
    instructions: Optional[str] = Field(
        None,
        description="Natural language voice description (overrides `voice` description for Qwen3 VoiceDesign / Maya-1).",
    )
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speaking speed multiplier (1.0 = pure unprocessed neural audio).")


class SceneLineItem(BaseModel):
    """A single dialogue turn in a multi-character scene."""
    speaker: str = Field(
        ...,
        description="Character name (e.g., 'Torvin', 'Nyx', 'Guard'). If already in Character Locker, reuses their locked voice; otherwise auto-creates and locks their voice!",
    )
    text: str = Field(
        ...,
        description="Dialogue spoken by this character. Supports Maya-1 inline tags (<laugh>, <sigh>, <whisper>, etc.) when `model_id='maya1'`.",
    )
    instruct: Optional[str] = Field(
        None,
        description="Voice description if this is a brand-new character (or if `model_id='maya1'`). Can also be defined once in `cast`.",
    )
    model_id: Optional[str] = Field(
        None,
        description="Optional override engine for this specific line ('qwen3_clone', 'qwen3_voicedesign', or 'maya1'). Defaults to 'qwen3_clone' so locked characters stay 100% consistent.",
    )
    speed: Optional[float] = Field(
        None,
        ge=0.5,
        le=2.0,
        description="Optional speaking speed override for this line.",
    )
    temperature: Optional[float] = Field(
        None,
        description="Optional native sampling temperature override for this line (e.g., 0.45 for ultra-consistent, 0.75 for wild emotional inflection).",
    )
    top_p: Optional[float] = Field(
        None,
        description="Optional native nucleus sampling top_p override for this line.",
    )
    repetition_penalty: Optional[float] = Field(
        None,
        description="Optional native repetition penalty override for this line.",
    )
    seed: Optional[int] = Field(
        None,
        description="Optional deterministic random seed override for this line.",
    )
    pause_after_ms: Optional[int] = Field(
        None,
        ge=-2000,
        le=5000,
        description="Gap in milliseconds after this line before the next character speaks. Positive values add silence; negative values (e.g. -200) overlap/interrupt the next speaker!",
    )
    ref_audio_path: Optional[str] = Field(
        None,
        description="Optional local path to an audio recording (.wav, .m4a, .mp3, etc.) to immediately clone and lock as this character.",
    )
    context_reset: bool = Field(
        False,
        description="Don't feed this line the previous line's audio, so a sudden emotion jump (e.g. calm to shouting) lands at full strength. Later lines still chain from it.",
    )
    continuity_group: Optional[str] = Field(
        None,
        description="Lines sharing a group chain context into each other even when they use different voices (e.g. one person's emotion clips 'Tamair (Angry)' and 'Tamair (Sad)'), so emotion shifts carry over gradually. Defaults to the speaker.",
    )
    ref_text: Optional[str] = Field(
        None,
        description="Optional exact spoken transcript for `ref_audio_path`. If omitted, automatically transcribed via on-device Whisper.",
    )
    blend_with: Optional[str] = Field(
        None,
        description="Emotion blend: ID or name of a second voice of the same person (e.g. 'Tamair (Sad)'). Its clip is joined onto this voice's reference so the read lands between the two emotions. Qwen3 clone only.",
    )
    blend_strength: str = Field(
        "subtle",
        description="'subtle' keeps this voice's emotion in front (blend clip heard first); 'strong' lets the blend clip lead (heard last, which pulls hardest).",
    )


class RerollSceneLineRequest(BaseModel):
    """Surgically re-synthesize a single line in an existing scene and immediately re-stitch the master scene WAV + SRT subtitles."""
    line_index: int = Field(..., ge=1, description="1-based line index in `scene.segments` to re-roll.")
    text: Optional[str] = Field(None, description="Optional updated dialogue text for this line (defaults to existing text).")
    instruct: Optional[str] = Field(None, description="Optional updated voice/acting instruction.")
    model_id: Optional[str] = Field(None, description="Optional engine override ('qwen3_clone', 'qwen3_voicedesign', or 'maya1').")
    speed: Optional[float] = Field(None, ge=0.5, le=2.0, description="Optional speaking speed multiplier for this line.")
    pause_after_ms: Optional[int] = Field(None, ge=-2000, le=5000, description="Optional updated pause/overlap after this line in ms.")
    temperature: Optional[float] = Field(None, description="Optional sampling temperature override for a fresh vocal take.")
    top_p: Optional[float] = Field(None, description="Optional nucleus sampling top_p override.")
    repetition_penalty: Optional[float] = Field(None, description="Optional repetition penalty override.")
    seed: Optional[int] = Field(None, description="Optional random seed override (pass a new integer to force a different vocal inflection take).")
    blend_with: Optional[str] = Field(None, description="Emotion blend voice for this line. Omit to keep the line's current blend; pass '' to remove it.")
    blend_strength: Optional[str] = Field(None, description="'subtle' or 'strong'. Omit to keep the current setting.")


class SceneRenderRequest(BaseModel):
    """
    Multi-character Scene & Script Renderer for LLMs and Game Engines.
    Provide either:
      - `lines`: Structured list of `{speaker, text, instruct?, pause_after_ms?, temperature?, top_p?, seed?}` objects, OR
      - `script_text`: Plain-text script where each line is `Speaker Name (optional voice description): Dialogue text`
    Automatically reuses existing locked characters by name, or creates + locks brand-new characters on the fly,
    synthesizes every segment, and stitches them into a single master 24kHz `.wav` scene with natural turn-taking pauses!
    """
    title: str = Field("Game Cutscene", description="Title of the scene.")
    cast: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional dictionary mapping Character Name -> Voice Description string OR Audio Recording Path string OR Dict `{'audio': '/path/to/clip.wav', 'ref_text': '...'}` OR Dict `{'character_id': 'char_...'}` to cast an existing locked voice under this name.",
    )
    lines: List[SceneLineItem] = Field(
        default_factory=list,
        description="Structured list of dialogue turns in the scene.",
    )
    script_text: Optional[str] = Field(
        None,
        description=(
            "Optional raw script text to parse if `lines` is omitted. Format per line:\n"
            "`Torvin (gruff 60s Scottish dwarf blacksmith, deep raspy baritone): Keep your coin, traveler.`\n"
            "`Nyx (witty 20s female rogue, crisp British accent): Oh, come on Torvin, everyone has a price.`"
        ),
    )
    default_pause_ms: int = Field(
        280,
        ge=-2000,
        le=5000,
        description="Default silence gap (or negative overlap in ms) inserted between dialogue turns when stitching the scene.",
    )
    speed: float = Field(
        1.0,
        ge=0.5,
        le=2.0,
        description="Default speaking speed multiplier across the scene (1.0 = pure unprocessed neural output by default).",
    )
    trim_silence: bool = Field(
        False,
        description="Optional silence trimming (False = pure unprocessed neural output by default).",
    )
    temperature: Optional[float] = Field(
        None,
        description="Optional scene-wide default native sampling temperature.",
    )
    top_p: Optional[float] = Field(
        None,
        description="Optional scene-wide default native top_p.",
    )
    repetition_penalty: Optional[float] = Field(
        None,
        description="Optional scene-wide default native repetition penalty.",
    )
    seed: Optional[int] = Field(
        None,
        description="Optional scene-wide default deterministic seed.",
    )
    auto_lock_new_characters: bool = Field(
        True,
        description="Automatically lock any newly designed character into the persistent Character Voice Locker so their voice stays identical across the rest of the scene and future scenes.",
    )
    context_continuity: bool = Field(
        True,
        description=(
            "Enable Rolling Audio+Text Context Chaining: each line sees both the character's permanent Golden Reference "
            "AND the immediately preceding line's audio + transcript so pitch, cadence, and conversational flow carry over naturally!"
        ),
    )
    natural_pacing: bool = Field(
        True,
        description=(
            "Vary the gap after lines that have no explicit `pause_after_ms`: quick replies, a beat after questions and "
            "emotion shifts, longer after trailing-off '...', and an overlap when a line is cut off with '—'. Scaled by `default_pause_ms`."
        ),
    )
    takes_per_line: int = Field(
        1,
        ge=1,
        le=4,
        description="Generate this many takes of every cloned line and keep the best (Whisper word match, plausible length, pitch close to the voice). Costs N x render time.",
    )


class BatchSceneRenderRequest(BaseModel):
    """Render an entire batch of game skits / cutscenes in a single low-friction pipeline call."""
    scenes: List[SceneRenderRequest] = Field(..., description="List of multi-character scenes/skits to render sequentially.")


class StitchTakesRequest(BaseModel):
    """Stitch an arbitrary list of existing `take_ids` into a single combined `.wav` scene."""
    title: str = Field("Stitched Scene", description="Title for the stitched scene.")
    take_ids: List[str] = Field(..., description="Ordered list of `take_id` strings to stitch together.")
    pause_ms: int = Field(280, ge=0, le=5000, description="Silence gap in milliseconds between takes.")


class OptimizePromptRequest(BaseModel):
    """Expand a brief or rough character concept into research-engineered Qwen3-TTS (7-Dimension) and Maya-1 (`<description>`) prompts."""
    concept: str = Field(
        ...,
        description="Short or rough character description (e.g., 'grumpy Scottish dwarf blacksmith', 'flirty board game rival', 'sinister cyberpunk AI').",
    )
    name: str = Field("Character", description="Optional character name.")
    model_id: str = Field("qwen3_voicedesign", description="Target engine ('qwen3_voicedesign' or 'maya1').")
    dialogue_text: str = Field("", description="Optional dialogue text to sanitize/optimize for the ear.")


# ============================================================================
# Core Discovery, Prompt Engineering & LLM System Guide Endpoints
# ============================================================================

@app.get("/api/llm-guide", tags=["System & Discovery"])
def api_llm_guide():
    """
    Returns the complete LLM System Prompt, Prompt Engineering Formulas (Qwen3 7-Dimension Formula,
    Maya-1 `<description>` template & 20 inline tags, Qwen3 Base 10-15s Golden Reference rules),
    and Skit Script syntax so any external LLM agent can self-configure in one GET call.
    """
    guide_path = BASE_DIR / "LLM_PROMPT_GUIDE.md"
    markdown_guide = guide_path.read_text(encoding="utf-8") if guide_path.exists() else ""
    return {
        "system_prompt_markdown": markdown_guide,
        "qwen3_voicedesign_formula": (
            "[1. Persona & Delivery Style] + [2. Gender & Age Bracket] + [3. Accent] + "
            "[4. Pitch & Register] + [5. Vocal Texture & Timbre] + [6. Pace & Rhythm] + [7. Studio Use Case]"
        ),
        "maya1_description_formula": (
            "[Archetype/Role], [Male/Female] voice in their [20s/30s/40s/50s/60s] with a [Accent] accent, "
            "[very low/low/medium/high] pitch, [gravelly/smooth/husky/clear/raspy] timbre, "
            "[brisk/conversational/measured/fast] pacing, [emotion] tone at [low/medium/high] intensity."
        ),
        "qwen3_clone_golden_reference_rules": {
            "optimal_duration_sec": "8 to 14 seconds (20-35 words)",
            "verbatim_ref_text": "Must match spoken reference audio 100% word-for-word with zero <tags>.",
        },
        "maya1_supported_tags": [t["tag"] for t in MAYA1_EMOTION_TAGS],
        "script_syntax_examples": [
            "Torvin (cynical 60s Scottish dwarf blacksmith, deep gravelly baritone): Keep your coin, traveler.",
            "Nyx (sharp witty 28yo female rogue, sleek British accent, medium-low husky pitch) [1.20x]: Oh, come on Torvin—",
            "[Overlap: 220ms]",
            "Torvin [1.25x]: Not for all the gold in the Citadel!",
            "Nyx [maya1, 1.20x]: <laugh> Fine, keep the sword. I only came for the map.",
        ],
    }


@app.post("/api/prompts/optimize", tags=["System & Discovery"])
def api_optimize_prompt(req: OptimizePromptRequest):
    """
    Auto-engineer a short character concept into high-performing Qwen3-TTS VoiceDesign (7-Dimension)
    and Maya-1 (`<description>`) prompts, plus a 10-second phonetically balanced Golden Reference sample.
    """
    bundle = optimize_character_prompts(req.concept, name=req.name)
    clean_text, extracted_tags = sanitize_dialogue_for_model(req.dialogue_text, model_id=req.model_id)
    compiled_active = compile_voice_instruct(req.concept, model_id=req.model_id, extracted_tags=extracted_tags)
    return {
        **bundle,
        "compiled_for_active_model": compiled_active,
        "sanitized_dialogue_text": clean_text,
        "extracted_tags": extracted_tags,
    }

@app.get("/api/health", tags=["System & Discovery"])
def api_health():
    """Lightweight health check for LLMs and game engines."""
    return {
        "ok": True,
        "active_model_id": engine.active_model_id,
        "status_message": engine.status_message,
    }


@app.get("/api/status", tags=["System & Discovery"])
def api_status():
    """Full studio state: active model, Metal VRAM usage, available models, Maya-1 tags, presets, characters, recent takes, and stitched scenes."""
    status = engine.get_status()
    status["presets"] = CHARACTER_PRESETS
    status["characters"] = _read_json(CHARACTERS_JSON, [])
    status["takes"] = _read_json(TAKES_JSON, [])[:40]
    status["scenes"] = _read_json(SCENES_JSON, [])[:30]
    return status


@app.get("/api/models", tags=["System & Discovery"])
def api_list_models():
    """List all supported local TTS models and which one is currently loaded in Apple Metal VRAM."""
    return {
        "active_model_id": engine.active_model_id,
        "models": list(MODEL_REGISTRY.values()),
    }


@app.get("/api/tags", tags=["System & Discovery"])
def api_list_maya_tags():
    """List all 20 inline paralinguistic emotion/acting tags supported by Maya-1."""
    return {"tags": MAYA1_EMOTION_TAGS}


@app.get("/api/presets", tags=["System & Discovery"])
def api_list_presets():
    """List built-in game character voice archetypes."""
    return {"presets": CHARACTER_PRESETS}


# ============================================================================
# Model Hot-Swapping Endpoints
# ============================================================================

@app.post("/api/model/load", tags=["Model Management"])
def api_load_model(req: LoadModelRequest):
    """Hot-swap the active TTS model in Apple Metal Unified Memory (unloads previous model automatically)."""
    try:
        return engine.load_model(req.model_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Speech Synthesis Endpoints (JSON Metadata + Direct WAV File Streaming)
# ============================================================================

def _resolve_character(character_id_or_name: str) -> Optional[dict]:
    if not character_id_or_name:
        return None
    target = character_id_or_name.strip().lower()
    chars = _read_json(CHARACTERS_JSON, [])
    # 1. Exact ID or exact name match
    for c in chars:
        if c["id"].lower() == target or c["name"].lower() == target:
            return c
    # 2. First-word / prefix match (e.g., "Torvin" matches "Torvin the Ironwright (Dwarven Blacksmith)")
    for c in chars:
        cname = c["name"].lower()
        first_word = cname.split()[0] if cname.split() else ""
        if target == first_word or cname.startswith(target + " ") or cname.startswith(target + " ("):
            return c
    return None


def _run_synthesis_and_save(req: SynthesizeRequest) -> dict:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Dialogue text cannot be empty.")

    ref_audio_path = None
    ref_text = None
    character_name = None
    resolved_char_id = req.character_id

    if req.character_id:
        char = _resolve_character(req.character_id)
        if not char and req.model_id == "qwen3_clone":
            raise HTTPException(status_code=404, detail=f"Character '{req.character_id}' not found in Character Locker.")
        if char:
            ref_audio_path = str(CHARACTERS_DIR / char["audio_file"])
            ref_text = char["ref_text"]
            character_name = char["name"]
            resolved_char_id = char["id"]
    elif req.model_id == "qwen3_clone":
        raise HTTPException(
            status_code=400,
            detail="Please provide `character_id` (or character name) from `/api/characters` when using `qwen3_clone`.",
        )

    blend_audio_path = None
    blend_text = None
    blend_name = None
    if req.blend_with:
        bchar = _resolve_character(req.blend_with)
        if not bchar:
            raise HTTPException(status_code=404, detail=f"Blend voice '{req.blend_with}' not found in the Voice Library.")
        if bchar["id"] != resolved_char_id:
            blend_audio_path = str(CHARACTERS_DIR / bchar["audio_file"])
            blend_text = bchar.get("ref_text")
            blend_name = bchar["name"]

    prev_audio_path = req.prev_audio_path
    prev_text = req.prev_text
    if req.prev_take_id and not prev_audio_path:
        takes = _read_json(TAKES_JSON, [])
        pt = next((t for t in takes if t["id"] == req.prev_take_id), None)
        if pt:
            prev_audio_path = str(TAKES_DIR / pt["filename"])
            prev_text = pt.get("text", "")

    try:
        audio_np, sr, metrics = engine.synthesize(
            model_id=req.model_id,
            text=req.text.strip(),
            instruct=req.instruct.strip(),
            language=req.language,
            speed=req.speed,
            trim_silence=req.trim_silence,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            seed=req.seed,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            prev_audio_path=prev_audio_path,
            prev_text=prev_text,
            dynamic_temperature=req.dynamic_temperature,
            blend_audio_path=blend_audio_path,
            blend_text=blend_text,
            blend_strength="strong" if req.blend_strength == "strong" else "subtle",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    take_id = f"take_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    filename = f"{take_id}.wav"
    filepath = TAKES_DIR / filename
    sf.write(str(filepath), audio_np, sr, subtype="PCM_16")

    take_record = {
        "id": take_id,
        "filename": filename,
        "file_path": str(filepath),
        "url": f"/audio/takes/{filename}",
        "model_id": req.model_id,
        "model_name": MODEL_REGISTRY[req.model_id]["name"],
        "text": req.text.strip(),
        "instruct": req.instruct.strip(),
        "character_id": resolved_char_id,
        "character_name": character_name,
        "sample_rate": sr,
        "metrics": metrics,
        "blend_with": blend_name if metrics.get("blended") else None,
        "created_at": time.strftime("%H:%M:%S"),
    }

    takes = _read_json(TAKES_JSON, [])
    takes.insert(0, take_record)
    _write_json(TAKES_JSON, takes[:100])
    return take_record


@app.post("/api/synthesize", tags=["Speech Synthesis"])
def api_synthesize(req: SynthesizeRequest):
    """
    Synthesize speech and return JSON metadata (including `take.id`, `take.url`, `take.file_path`, and speed metrics).
    """
    take_record = _run_synthesis_and_save(req)
    return {
        "take": take_record,
        "status": engine.get_status(),
    }


@app.post("/api/synthesize/wav", tags=["Speech Synthesis"])
def api_synthesize_wav(req: SynthesizeRequest):
    """
    Synthesize speech and return the binary 24kHz `.wav` audio file directly in the HTTP response body.
    Ideal for CLI scripts, game engines, or LLM pipelines that want immediate audio bytes.
    """
    take_record = _run_synthesis_and_save(req)
    return FileResponse(
        path=take_record["file_path"],
        media_type="audio/wav",
        filename=take_record["filename"],
        headers={"X-Take-Id": take_record["id"]},
    )


@app.post("/v1/audio/speech", tags=["Speech Synthesis"])
def openai_compatible_speech(req: OpenAISpeechRequest):
    """
    OpenAI-compatible TTS endpoint (`POST /v1/audio/speech`).
    - If `voice` matches a saved Character ID or Name, automatically uses `qwen3_clone`.
    - Otherwise uses `qwen3_voicedesign` or `maya1` with `instructions` (or `voice`) as the character description.
    """
    char = _resolve_character(req.voice)
    if char:
        synth_req = SynthesizeRequest(
            model_id="qwen3_clone",
            text=req.input,
            character_id=char["id"],
            speed=req.speed,
        )
    else:
        model_id = req.model if req.model in MODEL_REGISTRY else "qwen3_voicedesign"
        instruct = req.instructions or (
            req.voice if req.voice not in ("default", "alloy", "echo", "fable", "onyx", "nova", "shimmer")
            else "A warm, clear studio narrator voice with natural conversational pacing."
        )
        synth_req = SynthesizeRequest(
            model_id=model_id,
            text=req.input,
            instruct=instruct,
            speed=req.speed,
        )

    take_record = _run_synthesis_and_save(synth_req)
    return FileResponse(
        path=take_record["file_path"],
        media_type="audio/wav",
        filename=take_record["filename"],
        headers={"X-Take-Id": take_record["id"]},
    )


# ============================================================================
# Character Voice Locker & One-Shot LLM Workflow Endpoints
# ============================================================================

@app.get("/api/characters", tags=["Character Voice Locker"])
def api_list_characters():
    """List all locked NPC character voices available for zero-shot cloning."""
    return {"characters": _read_json(CHARACTERS_JSON, [])}


@app.post("/api/characters/design-and-lock", tags=["Character Voice Locker"])
def api_design_and_lock_character(req: DesignAndLockRequest):
    """
    All-in-one endpoint for LLMs:
    1. Generates a voice from a text description (`instruct`) using `qwen3_voicedesign` or `maya1`.
    2. Automatically locks that generated take into the Character Voice Locker under `name`.
    3. Returns the new `character` object (with `character.id`) so future calls can use `model_id='qwen3_clone'`.
    """
    if req.design_model_id not in ("qwen3_voicedesign", "maya1"):
        raise HTTPException(status_code=400, detail="design_model_id must be 'qwen3_voicedesign' or 'maya1'.")

    take = _run_synthesis_and_save(
        SynthesizeRequest(
            model_id=req.design_model_id,
            text=req.sample_text,
            instruct=req.instruct,
            speed=1.0,
            trim_silence=False,
            temperature=req.temperature,
            top_p=req.top_p,
            seed=req.seed,
        )
    )

    char_id = f"char_{int(time.time())}_{uuid.uuid4().hex[:5]}"
    dest_filename = f"{char_id}.wav"
    shutil.copy2(TAKES_DIR / take["filename"], CHARACTERS_DIR / dest_filename)

    char_record = {
        "id": char_id,
        "name": req.name.strip(),
        "description": req.instruct.strip(),
        "ref_text": take["text"],
        "audio_file": dest_filename,
        "file_path": str(CHARACTERS_DIR / dest_filename),
        "url": f"/audio/characters/{dest_filename}",
        "source_model": take["model_name"],
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }

    chars = _read_json(CHARACTERS_JSON, [])
    chars.insert(0, char_record)
    _write_json(CHARACTERS_JSON, chars)

    return {
        "character": char_record,
        "golden_take": take,
        "characters": chars,
    }


@app.post("/api/characters/from-take", tags=["Character Voice Locker"])
def api_save_character_from_take(req: SaveCharacterFromTakeRequest):
    """Lock an existing generated take (`take_id`) into the Character Voice Locker for consistent NPC cloning."""
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Character name is required.")

    takes = _read_json(TAKES_JSON, [])
    take = next((t for t in takes if t["id"] == req.take_id), None)
    if not take:
        raise HTTPException(status_code=404, detail="Take not found.")

    src_path = TAKES_DIR / take["filename"]
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="Take audio file missing on disk.")

    char_id = f"char_{int(time.time())}_{uuid.uuid4().hex[:5]}"
    dest_filename = f"{char_id}.wav"
    shutil.copy2(src_path, CHARACTERS_DIR / dest_filename)

    char_record = {
        "id": char_id,
        "name": req.name.strip(),
        "description": req.description.strip() or take.get("instruct", ""),
        "ref_text": take["text"],
        "audio_file": dest_filename,
        "file_path": str(CHARACTERS_DIR / dest_filename),
        "url": f"/audio/characters/{dest_filename}",
        "source_model": take.get("model_name", ""),
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }

    chars = _read_json(CHARACTERS_JSON, [])
    chars.insert(0, char_record)
    _write_json(CHARACTERS_JSON, chars)
    return {"character": char_record, "characters": chars}


def _find_optimal_reference_slice(audio_path: Union[str, Path]) -> Tuple[float, float]:
    """
    Pick a 6s-12.5s reference window that starts and ends inside natural pauses (so no word is cut in half
    and the Whisper transcript lines up with the audio), preferring the most intonation-rich window so the
    cloned voice inherits an expressive delivery rather than the flattest stretch of the recording.
    """
    sr = 24000
    try:
        cmd = ["ffmpeg", "-v", "error", "-i", str(audio_path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
        raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
        audio = np.frombuffer(raw, dtype=np.float32)
    except Exception:
        return 0.0, 10.0
    total_sec = audio.shape[0] / sr
    if total_sec <= 12.5:
        return 0.0, total_sec

    hop = int(sr * 0.02)
    n_frames = audio.shape[0] // hop
    rms = np.sqrt(np.mean(audio[: n_frames * hop].reshape(n_frames, hop) ** 2, axis=1))
    # Relative threshold so noisy phone recordings (which never reach -30 dBFS) still yield pauses
    quiet = rms < max(1e-4, float(np.percentile(rms, 20)) * 1.8, float(np.percentile(rms, 90)) * 0.06)

    # Candidate cut points: the middle of every quiet run of >= 200ms, plus the file edges
    cuts = [0.0]
    run_start = None
    for i, q in enumerate(np.append(quiet, False)):
        if q and run_start is None:
            run_start = i
        elif not q and run_start is not None:
            if i - run_start >= 10:
                cuts.append((run_start + i) / 2 * 0.02)
            run_start = None
    cuts.append(total_sec)

    candidates = []
    for i, t0 in enumerate(cuts):
        for t1 in cuts[i + 1:]:
            dur = t1 - t0
            if dur > 12.5:
                break
            if dur >= 6.0:
                candidates.append((t0, t1))

    if not candidates:
        # No usable pauses: end the window on the quietest frame between 9s and 12.5s
        lo, hi = int(9.0 / 0.02), min(n_frames, int(12.5 / 0.02))
        end_f = lo + int(np.argmin(rms[lo:hi])) if hi > lo else hi
        return 0.0, end_f * 0.02

    def _score(win: Tuple[float, float]) -> float:
        a = audio[int(win[0] * sr): int(win[1] * sr)]
        span = analyze_voice_acoustics(a, sr).get("pitch_semitone_span", 0.0)
        # Favour pitch movement, with a mild preference for ~10s windows (enough context for ICL)
        return float(span) - abs((win[1] - win[0]) - 10.0) * 0.4

    # Keep scoring cheap on long recordings
    if len(candidates) > 60:
        step = len(candidates) / 60.0
        candidates = [candidates[int(k * step)] for k in range(60)]
    best = max(candidates, key=_score)
    return round(best[0], 2), round(best[1], 2)


def _repair_gain_ramp(audio: np.ndarray, sr: int, max_head_sec: float = 4.0) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Some mics (notably the MacBook's built-in one when a browser opens it) come in 15-25 dB quiet for the
    first couple of seconds, then jump to full gain. Find that jump and lift everything before it to match.
    Returns (audio, None) untouched when there's no such jump, e.g. a performance that just starts quietly.
    """
    hop = max(1, int(sr * 0.02))
    n = audio.shape[0] // hop
    if n < 150:
        return audio, None
    db = 20 * np.log10(np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    smooth = np.convolve(db, np.ones(5) / 5, mode="same")  # 100 ms
    loud = db > np.percentile(db, 95) - 30
    body = db[int(2.5 / 0.02):][loud[int(2.5 / 0.02):]]
    if body.size < 50:
        return audio, None
    body_level = float(np.percentile(body, 80))
    start = int(0.3 / 0.02)
    cand = np.where(smooth[start: int(max_head_sec / 0.02)] > body_level - 10)[0]
    if cand.size == 0:
        return audio, None
    t_frame = start + int(cand[0])
    head = db[:t_frame]
    head_speech = head[head > np.percentile(head, 95) - 25] if head.size else head
    # A real gain ramp has words in the quiet part. A silent lead-in (room noise, a breath) is flat or too short.
    if t_frame < int(1.2 / 0.02) or head_speech.size < 25:
        return audio, None
    head_level = float(np.percentile(head_speech, 80))
    audible = head[head > -90]  # ignore digital silence
    if audible.size < 25 or float(np.percentile(audible, 90) - np.percentile(audible, 10)) < 12.0:
        return audio, None  # steady room noise, not syllables
    lift = body_level - head_level - 2.0
    if lift < 12.0:  # smaller differences are just a performance that starts softly
        return audio, None
    lift = min(lift, 30.0)
    t = t_frame * hop
    fade = min(int(0.06 * sr), t)
    g = 10 ** (lift / 20)
    gain = np.ones(audio.shape[0], dtype=np.float32)
    gain[: t - fade] = g
    gain[t - fade: t] = np.linspace(g, 1.0, fade, dtype=np.float32)
    out = (audio * gain).astype(np.float32)
    peak = float(np.max(np.abs(out)))
    if peak > 0.98:
        out *= 0.98 / peak
    return out, {"ramp_sec": round(t / sr, 2), "lift_db": round(lift, 1)}


def _repair_character_audio(char: dict) -> Optional[dict]:
    """
    Apply _repair_gain_ramp to a locked voice's reference clip. Always works from the untouched original
    (kept in characters/originals/), so running it again never stacks a second lift on top.
    """
    path = CHARACTERS_DIR / char["audio_file"]
    original = CHARACTERS_DIR / "originals" / char["audio_file"]
    audio, sr = sf.read(str(original if original.exists() else path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    fixed, info = _repair_gain_ramp(audio, sr)
    if not info:
        if original.exists():
            shutil.copy2(original, path)  # detection says there's nothing to fix: put the original back
        return None
    if not original.exists():
        original.parent.mkdir(exist_ok=True)
        shutil.copy2(path, original)
    sf.write(str(path), fixed, sr, subtype="PCM_16")
    return info


def _ingest_audio_recording(
    audio_source: Union[str, Path],
    name: str,
    ref_text: Optional[str] = None,
    description: Optional[str] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    repair_gain_ramp: bool = False,
) -> dict:
    """
    Ingests an external voice recording (.wav, .m4a, .mp3, etc.), extracts a pristine 24kHz mono Golden Reference,
    transcribes it via on-device Whisper if ref_text is omitted, and locks the character into the Voice Locker.
    """
    src = Path(str(audio_source).strip())
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"Audio recording not found at: {src}")

    duration = 10.0
    try:
        probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(src)]
        dur_out = subprocess.check_output(probe_cmd).decode().strip()
        duration = float(dur_out)
    except Exception:
        pass

    char_id = f"char_{int(time.time())}_{uuid.uuid4().hex[:5]}"
    dest_filename = f"{char_id}.wav"
    dest_path = CHARACTERS_DIR / dest_filename

    if start_sec is not None and end_sec is not None:
        cmd = ["ffmpeg", "-y", "-i", str(src), "-ss", str(start_sec), "-to", str(end_sec), "-ar", "24000", "-ac", "1", str(dest_path)]
    elif duration > 13.0:
        t0, t1 = _find_optimal_reference_slice(src)
        cmd = ["ffmpeg", "-y", "-i", str(src), "-ss", str(t0), "-to", str(t1), "-ar", "24000", "-ac", "1", str(dest_path)]
    else:
        cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "24000", "-ac", "1", str(dest_path)]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Fix a quiet start (mic gain still settling) before transcribing, so Whisper hears the first words too
    ramp_fix = None
    if repair_gain_ramp:
        try:
            ref_audio, ref_sr = sf.read(str(dest_path), dtype="float32")
            fixed, ramp_fix = _repair_gain_ramp(ref_audio if ref_audio.ndim == 1 else ref_audio.mean(axis=1), ref_sr)
            if ramp_fix:
                originals = CHARACTERS_DIR / "originals"
                originals.mkdir(exist_ok=True)
                shutil.copy2(dest_path, originals / dest_filename)
                sf.write(str(dest_path), fixed, ref_sr, subtype="PCM_16")
        except Exception:
            ramp_fix = None

    if not ref_text or not ref_text.strip():
        try:
            import mlx_audio.stt as stt
            whisper_model = stt.load_model("mlx-community/whisper-tiny")
            trans_res = whisper_model.generate(str(dest_path))
            ref_text = trans_res.text.strip()
        except Exception:
            ref_text = "Natural reference recording sample for voice cloning."

    char_record = {
        "id": char_id,
        "name": name.strip(),
        "description": description.strip() if description else f"Cloned voice from {src.name}",
        "ref_text": ref_text.strip(),
        "audio_file": dest_filename,
        "file_path": str(dest_path),
        "url": f"/audio/characters/{dest_filename}",
        "source_model": "User Recording (Cloned)",
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    if ramp_fix:
        char_record["gain_ramp_fixed"] = ramp_fix

    chars = _read_json(CHARACTERS_JSON, [])
    chars = [c for c in chars if c.get("name", "").strip().lower() != name.strip().lower() and c.get("id") != char_id]
    chars.insert(0, char_record)
    _write_json(CHARACTERS_JSON, chars)
    return char_record


@app.post("/api/characters/from-recording", tags=["Character Voice Locker"])
def api_create_character_from_recording(req: CharacterFromRecordingRequest):
    """Register an external voice recording (.wav, .m4a, .mp3, etc.) as a locked character voice with automatic 24kHz extraction & Whisper transcription."""
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Character name is required.")
    char_record = _ingest_audio_recording(
        audio_source=req.audio_path,
        name=req.name,
        ref_text=req.ref_text,
        description=req.description,
        start_sec=req.start_sec,
        end_sec=req.end_sec,
    )
    chars = _read_json(CHARACTERS_JSON, [])
    return {"character": char_record, "characters": chars}


@app.post("/api/characters/upload", tags=["Character Voice Locker"])
async def api_upload_character(
    name: str = Form(..., description="Character name"),
    ref_text: Optional[str] = Form(None, description="Exact transcript spoken in the reference audio (optional; transcribed via Whisper if omitted)"),
    description: str = Form("", description="Optional character notes"),
    file: UploadFile = File(..., description="Reference WAV/MP3/M4A audio clip"),
):
    """Upload an external reference audio clip to create a locked character voice."""
    if not name.strip():
        raise HTTPException(status_code=400, detail="Character name is required.")

    temp_id = f"upload_{int(time.time())}_{uuid.uuid4().hex[:5]}"
    ext = Path(file.filename or "ref.wav").suffix or ".wav"
    temp_path = DATA_DIR / f"{temp_id}{ext}"

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        char_record = _ingest_audio_recording(
            audio_source=temp_path,
            name=name,
            ref_text=ref_text,
            description=description,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()

    chars = _read_json(CHARACTERS_JSON, [])
    return {"character": char_record, "characters": chars}


@app.delete("/api/characters/{char_id}", tags=["Character Voice Locker"])
def api_delete_character(char_id: str):
    """Delete a character voice from the locker."""
    chars = _read_json(CHARACTERS_JSON, [])
    target = next((c for c in chars if c["id"] == char_id), None)
    if target:
        audio_path = CHARACTERS_DIR / target["audio_file"]
        if audio_path.exists():
            audio_path.unlink()
    chars = [c for c in chars if c["id"] != char_id]
    _write_json(CHARACTERS_JSON, chars)
    return {"characters": chars}


# ============================================================================
# Takes History Endpoints
# ============================================================================

@app.get("/api/takes", tags=["Takes History"])
def api_list_takes(limit: int = Query(40, ge=1, le=100)):
    """List recent generated audio takes."""
    return {"takes": _read_json(TAKES_JSON, [])[:limit]}


@app.delete("/api/takes/{take_id}", tags=["Takes History"])
def api_delete_take(take_id: str):
    """Delete a generated take and its WAV file."""
    takes = _read_json(TAKES_JSON, [])
    target = next((t for t in takes if t["id"] == take_id), None)
    if target:
        audio_path = TAKES_DIR / target["filename"]
        if audio_path.exists():
            audio_path.unlink()
    takes = [t for t in takes if t["id"] != take_id]
    _write_json(TAKES_JSON, takes)
    return {"takes": takes}


# ============================================================================
# Multi-Character Scene & Script Engine (Create/Reuse Cast + Stitch Audio)
# ============================================================================

def _parse_script_text_to_lines(script_text: str) -> List[SceneLineItem]:
    """
    Parse human or LLM plain-text script lines into structured `SceneLineItem` objects.
    Supports both multi-character scripts AND game NPC monologue/phone-call skits:
      - `Torvin (gruff 60s Scottish dwarf blacksmith): Keep your coin, traveler.`
      - `Nyx [maya1, 1.25x]: Oh, brilliant plan! <laugh> Let's walk right in.`
      - `[Pause: 600ms]`, `[pause 1.5s]`, or `[Overlap: 200ms]`
      - Game NPC Skit format:
          STORY: side_hustle_milk_1
          NPC TYPE: Normal (female, 20s)
          [casual]    "Don't worry I found a new side hustle"
          [pause 1.5s]
          [defensive] "No this isn't like last time"
    """
    parsed: List[SceneLineItem] = []
    line_pattern = re.compile(r"^\s*([^\(:\[]+?)(?:\s*\(([^\)]+)\))?(?:\s*\[([^\]]+)\])?\s*:\s*(.+)$")
    pause_pattern = re.compile(
        r"^\s*\[?\s*(pause|overlap)\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*(ms|s|sec|seconds?)?\s*\]?\s*$",
        re.IGNORECASE,
    )
    bracket_quote_pattern = re.compile(r"^\s*\[([^\]]+)\]\s*\"?(.+?)\"?\s*$")

    default_speaker = "NPC"
    default_npc_desc = None

    for raw_line in script_text.splitlines():
        s = raw_line.strip()
        if not s or s.startswith("#") or s.startswith("//") or set(s).issubset({"─", "-", "=", "_", " "}):
            continue

        # Header metadata lines for Game NPC Skit format
        upper = s.upper()
        if upper.startswith("STORY:"):
            story_id = s.split(":", 1)[1].strip()
            if story_id:
                default_speaker = story_id
            continue
        if upper.startswith("NPC TYPE:"):
            default_npc_desc = s.split(":", 1)[1].strip()
            continue
        if upper.startswith("TRIGGER:"):
            continue

        pm = pause_pattern.match(s)
        if pm:
            if parsed:
                kind = pm.group(1).lower()
                raw_num = abs(float(pm.group(2)))
                unit = (pm.group(3) or "ms").lower()
                val_ms = int(round(raw_num * 1000)) if unit.startswith("s") else int(round(raw_num))
                parsed[-1].pause_after_ms = -val_ms if kind == "overlap" else val_ms
            continue

        # Format B: `[casual] "Don't worry I found a new side hustle"`
        bqm = bracket_quote_pattern.match(s)
        if bqm and ":" not in s.split("]", 1)[0]:
            emotion_tag = bqm.group(1).strip()
            dialogue = bqm.group(2).strip().strip('"').strip("“").strip("”")
            if dialogue:
                combined_instruct = f"{default_npc_desc}, {emotion_tag} tone" if default_npc_desc else f"{emotion_tag} tone"
                parsed.append(
                    SceneLineItem(
                        speaker=default_speaker,
                        text=dialogue,
                        instruct=combined_instruct,
                    )
                )
            continue

        m = line_pattern.match(s)
        if m:
            speaker = m.group(1).strip()
            instruct = (m.group(2) or "").strip() or None
            directives = (m.group(3) or "").strip().lower()
            dialogue = m.group(4).strip().strip('"')
            raw_directives = (m.group(3) or "").strip()

            # `[blend: Tamair (Sad)]` or `[blend: Tamair (Sad) strong]`: mix in a second emotion clip
            blend_with = None
            blend_strength = "subtle"
            bl_m = re.search(r"blend\s*(?:with)?\s*[:=]\s*([^,\]]+)", raw_directives, re.IGNORECASE)
            if bl_m:
                bl = bl_m.group(1).strip()
                st_m = re.search(r"\s+(strong|subtle)$", bl, re.IGNORECASE)
                if st_m:
                    blend_strength = st_m.group(1).lower()
                    bl = bl[: st_m.start()].strip()
                blend_with = bl or None
                # Keep the voice name out of the other directives ("Clone", "Fast"... in a name)
                directives = re.sub(r"blend\s*(?:with)?\s*[:=]\s*[^,\]]+", "", directives, flags=re.IGNORECASE)

            model_id = None
            speed = None
            pause_after_ms = None
            temperature = None
            top_p = None
            repetition_penalty = None
            seed = None
            ref_audio_path = None
            ref_text = None

            full_meta = f"{instruct or ''} {directives or ''}"
            ref_m = re.search(r"ref\s*[:=]\s*([^\s,\]\)]+)", full_meta, re.IGNORECASE)
            if ref_m:
                ref_audio_path = ref_m.group(1).strip()
            ref_t_m = re.search(r"text\s*[:=]\s*([\"'][^\"']+[\"']|[^\],]+)", full_meta, re.IGNORECASE)
            if ref_t_m:
                ref_text = ref_t_m.group(1).strip().strip('"').strip("'")

            if directives:
                if "maya" in directives:
                    model_id = "maya1"
                elif "clone" in directives:
                    model_id = "qwen3_clone"
                elif "design" in directives or "qwen" in directives:
                    model_id = "qwen3_voicedesign"

                spd_m = re.search(r"(\d+(?:\.\d+)?)\s*x", directives)
                if spd_m:
                    speed = max(0.5, min(2.0, float(spd_m.group(1))))
                elif "fast" in directives:
                    speed = 1.25
                elif "slow" in directives:
                    speed = 0.92

                tmp_m = re.search(r"temp(?:erature)?\s*[:=]?\s*(\d+(?:\.\d+)?)", directives)
                if tmp_m:
                    temperature = float(tmp_m.group(1))
                tp_m = re.search(r"top_p\s*[:=]?\s*(\d+(?:\.\d+)?)", directives)
                if tp_m:
                    top_p = float(tp_m.group(1))
                rp_m = re.search(r"rep(?:etition)?(?:_penalty)?\s*[:=]?\s*(\d+(?:\.\d+)?)", directives)
                if rp_m:
                    repetition_penalty = float(rp_m.group(1))
                sd_m = re.search(r"seed\s*[:=]?\s*(\d+)", directives)
                if sd_m:
                    seed = int(sd_m.group(1))

                ov_m = re.search(r"overlap\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(ms|s)?", directives)
                if ov_m:
                    num = float(ov_m.group(1))
                    u = (ov_m.group(2) or "ms").lower()
                    pause_after_ms = -abs(int(num * 1000) if u == "s" else int(num))
                ps_m = re.search(r"pause\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*(ms|s)?", directives)
                if ps_m:
                    num = float(ps_m.group(1))
                    u = (ps_m.group(2) or "ms").lower()
                    pause_after_ms = int(num * 1000) if u == "s" else int(num)

            if dialogue:
                parsed.append(
                    SceneLineItem(
                        speaker=speaker,
                        text=dialogue,
                        instruct=instruct,
                        model_id=model_id,
                        speed=speed,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        seed=seed,
                        pause_after_ms=pause_after_ms,
                        ref_audio_path=ref_audio_path,
                        ref_text=ref_text,
                        blend_with=blend_with,
                        blend_strength=blend_strength,
                    )
                )

    return parsed


def _find_preset_instruct(speaker_name: str, for_maya: bool = False) -> Optional[str]:
    target = speaker_name.strip().lower()
    for p in CHARACTER_PRESETS:
        pname = p["name"].lower()
        first_word = pname.split()[0] if pname.split() else ""
        if target == pname or target == first_word or pname.startswith(target + " "):
            return p["instruct_maya"] if for_maya else p["instruct_qwen"]
    return None


def _format_srt_timestamp(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    hours = ms_total // 3_600_000
    ms_total %= 3_600_000
    minutes = ms_total // 60_000
    ms_total %= 60_000
    secs = ms_total // 1000
    millis = ms_total % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


LINE_TARGET_DBFS = -20.0   # speech level every line is pulled towards
LINE_LEVEL_KEEP = 0.15     # fraction of a line's own loudness difference kept, so whispers stay a bit quieter


# Natural dynamics: a whisper sits under a normal line and a shout sits above it, the way a real
# recording (or ElevenLabs output) keeps them, instead of every line being flattened to one level.
_LEVEL_OFFSETS_DB = [
    (("whisper", "hushed", "murmur"), -7.0),
    (("shout", "scream", "yell", "furious", "rage"), 3.0),
    (("angry", "excited", "panick", "frantic", "thrilled"), 1.5),
    (("sad", "tired", "scared", "afraid", "nervous", "sorrow", "quiet", "soft"), -2.0),
]


def _emotion_level_offset_db(seg: dict) -> float:
    """dB offset from the line's emotion label (e.g. 'Tamair (Whispering)') and how its text reads."""
    # Only the emotion part of the label counts ("Tamair (Sad)"), so a name like "Sadie" doesn't
    label = " ".join(re.findall(r"\(([^)]*)\)", seg.get("speaker", "")) + [seg.get("emotion") or ""]).lower()
    for words, db in _LEVEL_OFFSETS_DB:
        if re.search(r"\b(?:" + "|".join(words) + ")", label):
            return db
    text = re.sub(r"<[^>]+>", "", seg.get("text", ""))
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return 3.0  # ALL CAPS reads as shouted
    if text.count("!") >= 2:
        return 1.5
    if text.strip().endswith(("...", "…")) and "!" not in text:
        return -1.0  # trailing off
    return 0.0


def _level_line(audio: np.ndarray, sample_rate: int, offset_db: float = 0.0) -> np.ndarray:
    """Pull a line's speech loudness towards a common target (plus an emotion offset) so volume doesn't jump between takes."""
    if audio.size == 0:
        return audio
    hop = max(1, int(sample_rate * 0.02))
    n = audio.shape[0] // hop
    if n < 5:
        return audio
    rms = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1))
    level_db = 20 * np.log10(float(np.percentile(rms, 90)) + 1e-9)
    if level_db < -60:
        return audio
    gain_db = float(np.clip((LINE_TARGET_DBFS + offset_db - level_db) * (1 - LINE_LEVEL_KEEP), -15.0, 15.0))
    peak = float(np.max(np.abs(audio)))
    max_gain_db = 20 * np.log10(0.95 / peak) if peak > 0 else gain_db
    return (audio * 10 ** (min(gain_db, max_gain_db) / 20)).astype(np.float32)


def _trim_edges(audio: np.ndarray, sample_rate: int, lead_ms: int = 40, tail_ms: int = 90) -> np.ndarray:
    """Cut the dead air TTS leaves before/after a line (it varies 0-0.6s) so scene gaps are what we ask for."""
    hop = max(1, int(sample_rate * 0.01))
    n = audio.shape[0] // hop
    if n < 10:
        return audio
    db = 20 * np.log10(np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    voiced = np.where(db > db.max() - 35)[0]
    if voiced.size == 0:
        return audio
    start = max(0, voiced[0] * hop - int(sample_rate * lead_ms / 1000))
    end = min(audio.shape[0], (voiced[-1] + 1) * hop + int(sample_rate * tail_ms / 1000))
    out = audio[start:end].copy()
    fade = min(int(sample_rate * 0.01), out.shape[0] // 4)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    return out


def _mix_segments_into_master(segment_records: List[dict], sample_rate: int = 24000, level_lines: bool = True, trim_edges: bool = True) -> np.ndarray:
    """
    Mix and stitch ordered segment WAV files into a single master 24kHz timeline.
    Supports both positive turn-taking pauses (`pause_after_ms >= 0`) AND negative
    interruption overlaps (`pause_after_ms < 0`) so characters can cut each other off naturally!
    Updates `start_sec` and `end_sec` in place on each segment record.
    """
    loaded_waves: List[np.ndarray] = []
    starts_samples: List[int] = []
    cursor_samples = 0

    for idx, seg in enumerate(segment_records):
        take_filename = Path(seg["take_url"]).name
        wav_path = TAKES_DIR / take_filename
        if not wav_path.exists():
            audio = np.zeros(int(sample_rate * 0.2), dtype=np.float32)
        else:
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

        if trim_edges:
            audio = _trim_edges(audio, sample_rate)
        if level_lines:
            audio = _level_line(audio, sample_rate, _emotion_level_offset_db(seg))

        start_sample = max(0, cursor_samples)
        end_sample = start_sample + audio.shape[0]
        starts_samples.append(start_sample)
        loaded_waves.append(audio)

        seg["duration_sec"] = round(audio.shape[0] / sample_rate, 2)
        seg["start_sec"] = round(start_sample / sample_rate, 2)
        seg["end_sec"] = round(end_sample / sample_rate, 2)

        pause_ms = int(seg.get("pause_after_ms", 0) or 0) if idx < len(segment_records) - 1 else 0
        seg["pause_after_ms"] = pause_ms
        gap_samples = int(sample_rate * (pause_ms / 1000.0))
        # Ensure the next line never starts before the current line's start + 100ms
        cursor_samples = max(start_sample + int(sample_rate * 0.1), end_sample + gap_samples)

    total_samples = 0
    for start_s, wav in zip(starts_samples, loaded_waves):
        total_samples = max(total_samples, start_s + wav.shape[0])

    if total_samples == 0:
        return np.zeros(int(sample_rate * 0.25), dtype=np.float32)

    master = np.zeros(total_samples, dtype=np.float32)
    for start_s, wav in zip(starts_samples, loaded_waves):
        master[start_s : start_s + wav.shape[0]] += wav

    peak = float(np.max(np.abs(master)))
    if peak > 0.96:
        master = (master / peak) * 0.94

    return master.astype(np.float32)


def _write_scene_package_files(scene_record: dict) -> None:
    """
    Generate companion `.srt` subtitles and `_manifest.json` game-engine dialogue metadata
    alongside the stitched master `.wav` file for zero-friction Unity / Unreal / Godot import.
    """
    scene_id = scene_record["id"]
    srt_lines: List[str] = []
    for seg in scene_record.get("segments", []):
        idx = seg["index"]
        t0 = _format_srt_timestamp(seg["start_sec"])
        t1 = _format_srt_timestamp(seg["end_sec"])
        clean_text = re.sub(r"<[^>]+>", "", seg["text"]).strip() or seg["text"]
        srt_lines.append(f"{idx}\n{t0} --> {t1}\n{seg['speaker']}: {clean_text}\n")

    srt_filename = f"{scene_id}.srt"
    srt_path = SCENES_DIR / srt_filename
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

    manifest_filename = f"{scene_id}_manifest.json"
    manifest_path = SCENES_DIR / manifest_filename

    scene_record["srt_url"] = f"/audio/scenes/{srt_filename}"
    scene_record["manifest_url"] = f"/audio/scenes/{manifest_filename}"
    scene_record["bundle_zip_url"] = f"/api/scenes/{scene_id}/export.zip"

    manifest_payload = {
        "scene_id": scene_id,
        "title": scene_record["title"],
        "sample_rate": 24000,
        "duration_sec": scene_record["duration_sec"],
        "line_count": scene_record["line_count"],
        "master_audio_file": scene_record["filename"],
        "subtitles_file": srt_filename,
        "cast": scene_record.get("cast", []),
        "dialogue_lines": [
            {
                "index": seg["index"],
                "speaker": seg["speaker"],
                "character_id": seg.get("character_id"),
                "text": seg["text"],
                "clean_subtitle_text": re.sub(r"<[^>]+>", "", seg["text"]).strip() or seg["text"],
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "duration_sec": seg["duration_sec"],
                "pause_after_ms": seg.get("pause_after_ms", 0),
                "stem_file": f"lines/{seg['index']:03d}_{re.sub(r'[^a-zA-Z0-9_-]', '_', seg['speaker'].lower())}.wav",
                "take_id": seg["take_id"],
                "take_url": seg["take_url"],
            }
            for seg in scene_record.get("segments", [])
        ],
    }
    _write_json(manifest_path, manifest_payload)


_REF_PITCH_CACHE: Dict[str, float] = {}


def _ref_pitch(char: dict) -> float:
    if char["id"] not in _REF_PITCH_CACHE:
        try:
            a, sr = sf.read(str(CHARACTERS_DIR / char["audio_file"]), dtype="float32")
            _REF_PITCH_CACHE[char["id"]] = float(analyze_voice_acoustics(a if a.ndim == 1 else a.mean(axis=1), sr).get("pitch_hz_median", 0.0))
        except Exception:
            _REF_PITCH_CACHE[char["id"]] = 0.0
    return _REF_PITCH_CACHE[char["id"]]


def _score_take(take: dict, text: str, char: dict) -> Tuple[float, dict]:
    """Higher is better: words match the script, length is plausible for the text, pitch stays near the voice."""
    import difflib

    try:
        heard = _transcribe(TAKES_DIR / take["filename"])
    except Exception:
        heard = ""
    want = _norm_words(text)
    match = difflib.SequenceMatcher(None, want, _norm_words(heard)).ratio() if want else 1.0
    n_chars = max(1, len(re.sub(r"<[^>]+>", "", text).strip()))
    ms_per_char = take["metrics"]["duration_sec"] * 1000 / n_chars
    length_penalty = max(0.0, ms_per_char - 130) / 100 + max(0.0, 40 - ms_per_char) / 40
    ref_hz, hz = _ref_pitch(char), float(take["metrics"].get("pitch_hz_median", 0.0) or 0.0)
    # Pitch only counts once it drifts >~10% from the reference clip, and never outweighs getting the words right
    pitch_penalty = max(0.0, abs(np.log2(hz / ref_hz)) - 0.15) * 0.4 if ref_hz > 0 and hz > 0 else 0.0
    # Liveliness: among takes that say the words right, prefer the one whose pitch actually moves.
    # Flat, monotone reads are the main thing that separates local TTS from ElevenLabs-style output.
    span = float(take["metrics"].get("pitch_semitone_span", 0.0) or 0.0)
    emotional = _emotion_temperature_boost(text) >= 0.05
    expressive_bonus = (0.3 if emotional else 0.12) * min(span, 14.0) / 14.0
    if span < 2.5 and len(want) >= 4:
        expressive_bonus -= 0.15  # monotone
    score = 2.0 * match - length_penalty - pitch_penalty + expressive_bonus
    return score, {"match": round(match, 2), "ms_per_char": round(ms_per_char), "pitch_hz": round(hz),
                   "pitch_span": round(span, 1), "score": round(score, 3)}


def _pick_best_take(candidates: List[dict], text: str, char: dict) -> dict:
    scored = [(_score_take(t, text, char), t) for t in candidates]
    (best_score, best_info), best = max(scored, key=lambda x: x[0][0])
    best = dict(best)
    best["candidates"] = [{"take_id": t["id"], "url": t["url"], **info} for (_, info), t in scored]
    return best


_CALM_EMOTIONS = ("neutral", "tired", "sad", "whisper", "confused")


def _natural_gap_ms(line: SceneLineItem, nxt: SceneLineItem, base_ms: int) -> int:
    """Gap after `line` based on how it ends and what comes next, scaled so base_ms=280 gives the defaults."""
    text = line.text.strip()
    scale = max(0.0, base_ms / 280.0)
    same_person = _continuity_key(line) == _continuity_key(nxt)
    if text.endswith(("—", "–", "--", "-")):
        return -180  # cut off: next speaker barges in
    if same_person and line.speaker.strip().lower() == nxt.speaker.strip().lower():
        gap = 220  # same voice carrying on
    elif same_person:
        gap = 480  # same person, new emotion: a beat to shift gear
    else:
        gap = 300
    if text.endswith(("...", "…")):
        gap += 400
    elif text.endswith("?"):
        gap += 60
    elif text.endswith("!"):
        gap -= 60
    if len(_norm_words(text)) > 16:
        gap += 120  # a breath after a long line
    nxt_text = nxt.text.strip()
    letters = [c for c in nxt_text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        gap -= 120  # shouted reply comes in hot
    return int(round(gap * scale))


def _continuity_key(line: SceneLineItem) -> str:
    return (line.continuity_group or line.speaker).strip().lower()


def _execute_scene_render(req: SceneRenderRequest) -> dict:
    """
    2-Pass Smart Model-Swap Scene Director:
      - Pass 1 (New Character Discovery & Voice Locking): Synthesizes the first line of any brand-new
        character using `qwen3_voicedesign` (or `maya1`), locks their voice into `CHARACTERS_JSON`,
        and caches that first line's audio take so it isn't synthesized twice.
      - Pass 2 (Zero-Swap Batch Dialogue Rendering): Groups all remaining lines by engine (`qwen3_clone` vs `maya1`)
        so a 15-line skit between 4 characters swaps models at most ONCE instead of alternating on every line!
    """
    t_start = time.perf_counter()
    raw_lines = list(req.lines)
    if not raw_lines and req.script_text:
        raw_lines = _parse_script_text_to_lines(req.script_text)

    lines = [l for l in raw_lines if l.text and l.text.strip()]
    if not lines:
        raise HTTPException(
            status_code=400,
            detail="Provide either `lines` (list of speaker+text turns) or `script_text` formatted as `Speaker (optional voice desc): Dialogue`.",
        )

    # Normalize cast lookup (case-insensitive)
    cast_descriptions: Dict[str, str] = {}
    for k, v in (req.cast or {}).items():
        s_key = k.strip().lower()
        if isinstance(v, str):
            cast_descriptions[s_key] = v.strip()
        elif isinstance(v, dict):
            if v.get("description"):
                cast_descriptions[s_key] = str(v["description"]).strip()
            elif v.get("instruct"):
                cast_descriptions[s_key] = str(v["instruct"]).strip()

    for item in lines:
        if item.instruct and item.speaker.strip().lower() not in cast_descriptions:
            cast_descriptions[item.speaker.strip().lower()] = item.instruct.strip()

    pre_rendered_takes: Dict[int, tuple] = {}
    resolved_cast_map: Dict[str, dict] = {}
    cast_summary: Dict[str, dict] = {}
    model_swaps = 0
    last_used_model = engine.active_model_id

    # -------------------------------------------------------------------------
    # PASS 0: Ingest external audio recordings provided in `cast` or `lines`
    # -------------------------------------------------------------------------
    for raw_speaker, cast_val in (req.cast or {}).items():
        s_key = raw_speaker.strip().lower()
        if isinstance(cast_val, dict) and cast_val.get("character_id"):
            char = _resolve_character(str(cast_val["character_id"]))
            if not char:
                raise HTTPException(status_code=404, detail=f"Voice '{cast_val['character_id']}' for {raw_speaker} not found in Character Locker.")
            resolved_cast_map[s_key] = char
        elif isinstance(cast_val, dict) and any(k in cast_val for k in ("audio", "ref_audio", "file", "audio_path")):
            audio_src = cast_val.get("audio") or cast_val.get("ref_audio") or cast_val.get("file") or cast_val.get("audio_path")
            ref_txt = cast_val.get("ref_text") or cast_val.get("text")
            desc = cast_val.get("description") or f"Cloned voice from audio recording for {raw_speaker}"
            char = _ingest_audio_recording(audio_src, name=raw_speaker, ref_text=ref_txt, description=desc)
            resolved_cast_map[s_key] = char
        elif isinstance(cast_val, str):
            cv = cast_val.strip()
            audio_extensions = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac")
            is_file = any(cv.lower().endswith(ext) for ext in audio_extensions) or Path(cv).is_file()
            if is_file:
                char = _ingest_audio_recording(cv, name=raw_speaker, description=f"Cloned voice from {Path(cv).name}")
                resolved_cast_map[s_key] = char

    for item in lines:
        s_key = item.speaker.strip().lower()
        if item.ref_audio_path and s_key not in resolved_cast_map:
            char = _ingest_audio_recording(
                item.ref_audio_path,
                name=item.speaker.strip(),
                ref_text=item.ref_text,
                description=item.instruct or f"Cloned voice from {Path(item.ref_audio_path).name}",
            )
            resolved_cast_map[s_key] = char

    # -------------------------------------------------------------------------
    # PASS 1: Discover brand-new characters, batch by design engine, and lock!
    # -------------------------------------------------------------------------
    new_speaker_first_line: Dict[str, int] = {}
    for idx, line in enumerate(lines):
        speaker = line.speaker.strip()
        s_key = speaker.lower()
        if s_key in resolved_cast_map:
            continue
        existing_char = _resolve_character(speaker)
        if existing_char is not None:
            resolved_cast_map[s_key] = existing_char
        elif s_key not in new_speaker_first_line:
            new_speaker_first_line[s_key] = idx

    # Sort new character discovery lines by design engine so we don't swap back and forth
    def _design_engine_for_line(idx: int) -> str:
        l = lines[idx]
        if l.model_id in ("maya1", "qwen3_voicedesign"):
            return l.model_id
        return "maya1" if ("<" in l.text and ">" in l.text) else "qwen3_voicedesign"

    discovery_indices = sorted(new_speaker_first_line.values(), key=_design_engine_for_line)
    for idx in discovery_indices:
        line = lines[idx]
        speaker = line.speaker.strip()
        s_key = speaker.lower()
        design_engine = _design_engine_for_line(idx)
        line_speed = line.speed if line.speed is not None else req.speed
        instruct = (
            line.instruct
            or cast_descriptions.get(s_key)
            or _find_preset_instruct(speaker, for_maya=(design_engine == "maya1"))
            or f"Distinctive studio game character voice for {speaker}, clear diction, natural conversational pacing."
        )

        first_line_text = line.text.strip()
        has_inline_tags = "<" in first_line_text and ">" in first_line_text
        # Qwen3-TTS Base ICL cloning works best with an 8-14s (50-220 char) tag-free reference clip.
        # If the character's first line in the skit is too short (<50 chars) or has <tags>,
        # synthesize a dedicated 9-second Golden Reference Anchor to lock their voice cleanly!
        needs_golden_anchor = req.auto_lock_new_characters and (len(first_line_text) < 50 or has_inline_tags)

        if needs_golden_anchor:
            anchor_engine = "qwen3_voicedesign"
            if anchor_engine != last_used_model:
                model_swaps += 1
                last_used_model = anchor_engine
            anchor_text = (
                "Keep your eyes sharp, friend. Nothing in this place is what it seems, "
                "and if we want to make it out before sunrise, you had better stick right beside me."
            )
            anchor_take = _run_synthesis_and_save(
                SynthesizeRequest(
                    model_id=anchor_engine,
                    text=anchor_text,
                    instruct=instruct,
                    speed=1.0,
                    trim_silence=False,
                    temperature=line.temperature if line.temperature is not None else req.temperature,
                    top_p=line.top_p if line.top_p is not None else req.top_p,
                    seed=line.seed if line.seed is not None else req.seed,
                )
            )
            locked_char = api_save_character_from_take(
                SaveCharacterFromTakeRequest(take_id=anchor_take["id"], name=speaker, description=instruct)
            )["character"]
            resolved_cast_map[s_key] = locked_char
            # Leave `idx` out of `pre_rendered_takes` so Pass 2 renders Line #1 using the locked voice!
            continue

        if design_engine != last_used_model:
            model_swaps += 1
            last_used_model = design_engine

        # When auto-locking from Line #1, always synthesize the reference cleanly at speed=1.0, trim_silence=False
        take = _run_synthesis_and_save(
            SynthesizeRequest(
                model_id=design_engine,
                text=first_line_text,
                instruct=instruct,
                speed=1.0 if req.auto_lock_new_characters else line_speed,
                trim_silence=False if req.auto_lock_new_characters else req.trim_silence,
                temperature=line.temperature if line.temperature is not None else req.temperature,
                top_p=line.top_p if line.top_p is not None else req.top_p,
                repetition_penalty=line.repetition_penalty if line.repetition_penalty is not None else req.repetition_penalty,
                seed=line.seed if line.seed is not None else req.seed,
            )
        )
        action = "designed_new_voice"
        locked_char = None
        if req.auto_lock_new_characters:
            locked_char = api_save_character_from_take(
                SaveCharacterFromTakeRequest(take_id=take["id"], name=speaker, description=instruct)
            )["character"]
            resolved_cast_map[s_key] = locked_char
            take["character_id"] = locked_char["id"]
            take["character_name"] = locked_char["name"]
            action = "created_and_locked"

        pre_rendered_takes[idx] = (take, action, locked_char, instruct, line_speed)

    # -------------------------------------------------------------------------
    # PASS 2: Batch-render all remaining lines grouped by engine (`qwen3_clone` vs `maya1`)
    # -------------------------------------------------------------------------
    remaining_indices = [i for i in range(len(lines)) if i not in pre_rendered_takes]

    def _target_engine_for_remaining(idx: int) -> str:
        l = lines[idx]
        s_key = l.speaker.strip().lower()
        has_locked = (s_key in resolved_cast_map) or (_resolve_character(l.speaker.strip()) is not None)
        # If the character is already locked in the Character Locker, ALWAYS use `qwen3_clone`
        # unless the line explicitly contains a Maya-1 vocal sound tag (<laugh>, <sigh>, <gasp>, etc.)
        # AND explicitly requests `model_id='maya1'`. This stops LLMs from accidentally drifting voices!
        has_maya_vocal_tag = any(
            t in l.text.lower()
            for t in ("<laugh", "<chuckle", "<giggle", "<sigh", "<gasp", "<gulp", "<snort", "<cry", "<scream", "<sing")
        )
        if has_locked:
            if l.model_id == "maya1" and has_maya_vocal_tag:
                return "maya1"
            return "qwen3_clone"
        if l.model_id == "maya1":
            return "maya1"
        if l.model_id == "qwen3_voicedesign":
            return "qwen3_voicedesign"
        return "maya1" if has_maya_vocal_tag else "qwen3_voicedesign"

    # Prefer rendering whichever engine is ALREADY loaded in VRAM first!
    def _sort_key_for_remaining(idx: int):
        eng = _target_engine_for_remaining(idx)
        return (0 if eng == last_used_model else 1, eng, idx)

    for idx in sorted(remaining_indices, key=_sort_key_for_remaining):
        line = lines[idx]
        speaker = line.speaker.strip()
        s_key = speaker.lower()
        target_eng = _target_engine_for_remaining(idx)
        line_speed = line.speed if line.speed is not None else req.speed
        line_temp = line.temperature if line.temperature is not None else req.temperature
        line_top_p = line.top_p if line.top_p is not None else req.top_p
        line_rep = line.repetition_penalty if line.repetition_penalty is not None else req.repetition_penalty
        line_seed = line.seed if line.seed is not None else req.seed
        existing_char = resolved_cast_map.get(s_key) or _resolve_character(speaker)

        # Rolling Audio+Text Context Chaining: find this character's most recent previous rendered line in the skit!
        prev_audio_path = None
        prev_text = None
        if req.context_continuity and not line.context_reset:
            group = _continuity_key(line)
            for prev_idx in range(idx - 1, -1, -1):
                if prev_idx in pre_rendered_takes and _continuity_key(lines[prev_idx]) == group:
                    prev_take_obj = pre_rendered_takes[prev_idx][0]
                    prev_audio_path = prev_take_obj.get("file_path") or str(TAKES_DIR / prev_take_obj["filename"])
                    prev_text = lines[prev_idx].text.strip()
                    break

        if target_eng != last_used_model:
            model_swaps += 1
            last_used_model = target_eng

        if target_eng == "qwen3_clone" and existing_char is not None:
            n_takes = 1 if line_seed is not None else req.takes_per_line
            candidates = []
            for k in range(n_takes):
                candidates.append(_run_synthesis_and_save(
                    SynthesizeRequest(
                        model_id="qwen3_clone",
                        character_id=existing_char["id"],
                        text=line.text.strip(),
                        speed=line_speed,
                        trim_silence=req.trim_silence,
                        temperature=line_temp,
                        top_p=line_top_p,
                        repetition_penalty=line_rep,
                        # Clone mode otherwise falls back to one stable seed per character, which would repeat the same take
                        seed=line_seed if n_takes == 1 else random.randint(1, 2**31 - 1),
                        prev_audio_path=prev_audio_path,
                        prev_text=prev_text,
                        blend_with=line.blend_with,
                        blend_strength=line.blend_strength,
                    )
                ))
            take = _pick_best_take(candidates, line.text, existing_char) if len(candidates) > 1 else candidates[0]
            action = "reused_locked_character_context_chained" if prev_audio_path else "reused_locked_character"
            instruct = existing_char.get("description", "")
        else:
            # Also enrich `instruct` with the immediately preceding line in the scene (even if spoken by another speaker!)
            scene_prev_context = ""
            if req.context_continuity and idx > 0:
                prev_line_item = lines[idx - 1]
                prev_clean = re.sub(r"<[^>]+>", "", prev_line_item.text).strip()
                if prev_clean:
                    scene_prev_context = f", responding directly to the previous line ('{prev_clean[:80]}')"

            base_instruct = (
                line.instruct
                or cast_descriptions.get(s_key)
                or (existing_char["description"] if existing_char else None)
                or _find_preset_instruct(speaker, for_maya=(target_eng == "maya1"))
                or f"Expressive character voice for {speaker}, clear diction, natural conversational pacing."
            )
            instruct = f"{base_instruct.rstrip('.')}{scene_prev_context}."
            take = _run_synthesis_and_save(
                SynthesizeRequest(
                    model_id=target_eng,
                    character_id=existing_char["id"] if existing_char else None,
                    text=line.text.strip(),
                    instruct=instruct,
                    speed=line_speed,
                    trim_silence=req.trim_silence,
                    temperature=line_temp,
                    top_p=line_top_p,
                    repetition_penalty=line_rep,
                    seed=line_seed,
                    prev_audio_path=prev_audio_path,
                    prev_text=prev_text,
                )
            )
            action = "reused_character_with_maya1" if existing_char else f"designed_with_{target_eng}"

        pre_rendered_takes[idx] = (take, action, existing_char, instruct, line_speed)

    # -------------------------------------------------------------------------
    # Assemble ordered segments, mix overlaps/pauses, and export package files
    # -------------------------------------------------------------------------
    segment_records: List[dict] = []
    for idx, line in enumerate(lines):
        speaker = line.speaker.strip()
        s_key = speaker.lower()
        take, action, char_obj, instruct_used, line_speed = pre_rendered_takes[idx]
        if line.pause_after_ms is not None:
            pause_ms = line.pause_after_ms
        elif req.natural_pacing and idx < len(lines) - 1:
            pause_ms = _natural_gap_ms(line, lines[idx + 1], req.default_pause_ms)
        else:
            pause_ms = req.default_pause_ms

        if speaker not in cast_summary or action == "created_and_locked":
            cast_summary[speaker] = {
                "speaker": speaker,
                "character_id": char_obj["id"] if char_obj else None,
                "status": action,
                "description": char_obj["description"] if char_obj else instruct_used,
            }

        segment_records.append({
            "index": idx + 1,
            "speaker": speaker,
            "continuity_group": _continuity_key(line),
            "character_id": char_obj["id"] if char_obj else None,
            "model_id": take["model_id"],
            "action": action,
            "context_chained": bool(take["metrics"].get("context_chained", False)),
            "blend_with": line.blend_with if take["metrics"].get("blended") else None,
            "blend_strength": line.blend_strength if take["metrics"].get("blended") else None,
            "text": line.text.strip(),
            "instruct": instruct_used,
            "speed": line_speed,
            "pitch_hz_median": take["metrics"].get("pitch_hz_median", 0.0),
            "voice_register": take["metrics"].get("voice_register", ""),
            "take_id": take["id"],
            "take_url": take["url"],
            "candidates": take.get("candidates", []),
            "duration_sec": take["metrics"]["duration_sec"],
            "start_sec": 0.0,
            "end_sec": 0.0,
            "pause_after_ms": pause_ms if idx < len(lines) - 1 else 0,
        })

    sample_rate = 24000
    master_audio = _mix_segments_into_master(segment_records, sample_rate=sample_rate)

    scene_id = f"scene_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    scene_filename = f"{scene_id}.wav"
    scene_filepath = SCENES_DIR / scene_filename
    sf.write(str(scene_filepath), master_audio, sample_rate, subtype="PCM_16")

    total_duration = round(master_audio.shape[0] / sample_rate, 2)
    elapsed_sec = round(time.perf_counter() - t_start, 2)

    scene_record = {
        "id": scene_id,
        "title": req.title.strip() or "Game Scene",
        "filename": scene_filename,
        "file_path": str(scene_filepath),
        "url": f"/audio/scenes/{scene_filename}",
        "duration_sec": total_duration,
        "elapsed_sec": elapsed_sec,
        "model_swaps": model_swaps,
        "line_count": len(segment_records),
        "cast": list(cast_summary.values()),
        "segments": segment_records,
        "created_at": time.strftime("%H:%M:%S"),
    }

    _write_scene_package_files(scene_record)

    scenes = _read_json(SCENES_JSON, [])
    scenes.insert(0, scene_record)
    _write_json(SCENES_JSON, scenes[:50])
    return scene_record


@app.post("/api/scenes/render", tags=["Multi-Character Scene Director"])
def api_render_scene(req: SceneRenderRequest):
    """
    Render a complete multi-character script into a stitched master `.wav` scene using 2-Pass Smart Model-Swap Batching:
    - Pass 1: Discovers any brand-new characters, designs their voice from `instruct`, and locks them into the Character Locker.
    - Pass 2: Batch-renders all locked character lines together with `qwen3_clone` to minimize VRAM model swaps.
    - Supports positive silence pauses (`pause_after_ms > 0`) AND negative interruption overlaps (`pause_after_ms < 0`).
    - Automatically exports `.srt` subtitles, `_manifest.json`, and a 1-click `.zip` Game Engine Cutscene Bundle.
    """
    scene_record = _execute_scene_render(req)
    return {
        "scene": scene_record,
        "characters": _read_json(CHARACTERS_JSON, []),
        "takes": _read_json(TAKES_JSON, [])[:40],
        "status": engine.get_status(),
    }


@app.post("/api/scenes/batch", tags=["Multi-Character Scene Director"])
def api_render_scenes_batch(req: BatchSceneRenderRequest):
    """
    Low-friction batch skit pipeline: render multiple game skits / cutscenes in a single API call.
    Characters created and locked in Skit #1 are automatically reused in Skit #2..N!
    """
    t0 = time.perf_counter()
    rendered_scenes = []
    for s_req in req.scenes:
        rendered_scenes.append(_execute_scene_render(s_req))
    return {
        "count": len(rendered_scenes),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
        "scenes": rendered_scenes,
        "characters": _read_json(CHARACTERS_JSON, []),
        "status": engine.get_status(),
    }


@app.post("/api/scenes/{scene_id}/reroll-line", tags=["Multi-Character Scene Director"])
def api_reroll_scene_line(scene_id: str, req: RerollSceneLineRequest):
    """
    Surgically re-roll (or edit the text/speed/pause/engine of) a single line in an existing scene
    without re-generating the rest of the skit! Immediately re-mixes the master `.wav` and updates `.srt` & `manifest.json`.
    """
    t0 = time.perf_counter()
    scenes = _read_json(SCENES_JSON, [])
    scene = next((s for s in scenes if s["id"] == scene_id), None)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found.")

    segments = scene.get("segments", [])
    target_seg = next((seg for seg in segments if seg["index"] == req.line_index), None)
    if not target_seg:
        raise HTTPException(status_code=404, detail=f"Line index {req.line_index} not found in scene '{scene_id}'.")

    speaker = target_seg["speaker"]
    new_text = (req.text if req.text is not None else target_seg["text"]).strip()
    new_speed = req.speed if req.speed is not None else float(target_seg.get("speed", 1.0))
    char_obj = _resolve_character(target_seg.get("character_id") or speaker)

    model_id = req.model_id or target_seg.get("model_id") or ("qwen3_clone" if char_obj else "qwen3_voicedesign")
    instruct = (
        req.instruct
        or target_seg.get("instruct")
        or (char_obj["description"] if char_obj else "")
        or f"Expressive game character voice for {speaker}."
    )

    # Find the character's preceding line in the scene so re-rolls preserve exact context continuity!
    prev_take_id = None
    for seg in reversed(segments):
        if seg["index"] < req.line_index and (seg.get("continuity_group") or seg["speaker"]).strip().lower() == (target_seg.get("continuity_group") or speaker).strip().lower():
            prev_take_id = seg.get("take_id")
            break

    synth_req = SynthesizeRequest(
        model_id=model_id,
        text=new_text,
        instruct=instruct,
        character_id=char_obj["id"] if char_obj else None,
        speed=new_speed,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        # Clone mode otherwise derives a fixed per-character seed, so a re-roll without a seed would
        # sample the exact same take again.
        seed=req.seed if req.seed is not None else random.randint(1, 2**31 - 1),
        prev_take_id=prev_take_id,
        blend_with=(req.blend_with if req.blend_with is not None else target_seg.get("blend_with")) or None if model_id == "qwen3_clone" else None,
        blend_strength=req.blend_strength or target_seg.get("blend_strength") or "subtle",
    )
    new_take = _run_synthesis_and_save(synth_req)
    target_seg["blend_with"] = new_take.get("blend_with")
    target_seg["blend_strength"] = synth_req.blend_strength if new_take.get("blend_with") else None

    target_seg["text"] = new_text
    target_seg["instruct"] = instruct
    target_seg["model_id"] = model_id
    target_seg["speed"] = new_speed
    target_seg["pitch_hz_median"] = new_take["metrics"].get("pitch_hz_median", 0.0)
    target_seg["voice_register"] = new_take["metrics"].get("voice_register", "")
    # Keep earlier takes switchable so a re-roll never loses a take you liked
    previous = target_seg.get("candidates") or [{"take_id": target_seg["take_id"], "url": target_seg["take_url"]}]
    target_seg["candidates"] = previous + [{"take_id": new_take["id"], "url": new_take["url"], "rerolled": True}]
    target_seg["take_id"] = new_take["id"]
    target_seg["take_url"] = new_take["url"]
    target_seg["action"] = f"rerolled_with_{model_id}"
    if req.pause_after_ms is not None:
        target_seg["pause_after_ms"] = req.pause_after_ms

    sample_rate = 24000
    master_audio = _mix_segments_into_master(segments, sample_rate=sample_rate)
    scene_filepath = SCENES_DIR / scene["filename"]
    sf.write(str(scene_filepath), master_audio, sample_rate, subtype="PCM_16")

    scene["duration_sec"] = round(master_audio.shape[0] / sample_rate, 2)
    scene["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    _write_scene_package_files(scene)
    _write_json(SCENES_JSON, scenes)

    return {
        "scene": scene,
        "rerolled_segment": target_seg,
        "new_take": new_take,
        "takes": _read_json(TAKES_JSON, [])[:40],
        "status": engine.get_status(),
    }


class UseTakeRequest(BaseModel):
    line_index: int = Field(..., ge=1, description="1-based line index in the scene.")
    take_id: str = Field(..., description="One of that line's `candidates` take ids.")


@app.post("/api/scenes/{scene_id}/use-take", tags=["Multi-Character Scene Director"])
def api_use_scene_take(scene_id: str, req: UseTakeRequest):
    """Swap a line to one of its other takes (best-of-N runners-up or earlier re-rolls) and re-stitch the scene."""
    scenes = _read_json(SCENES_JSON, [])
    scene = next((s for s in scenes if s["id"] == scene_id), None)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found.")
    seg = next((g for g in scene.get("segments", []) if g["index"] == req.line_index), None)
    if not seg:
        raise HTTPException(status_code=404, detail=f"Line {req.line_index} not found.")
    cand = next((c for c in seg.get("candidates", []) if c["take_id"] == req.take_id), None)
    take = next((t for t in _read_json(TAKES_JSON, []) if t["id"] == req.take_id), None)
    if not cand or not (TAKES_DIR / f"{req.take_id}.wav").exists():
        raise HTTPException(status_code=404, detail="That take is not available for this line any more.")
    seg["take_id"] = req.take_id
    seg["take_url"] = cand["url"]
    if take:
        seg["pitch_hz_median"] = take["metrics"].get("pitch_hz_median", 0.0)
    sample_rate = 24000
    master_audio = _mix_segments_into_master(scene["segments"], sample_rate=sample_rate)
    sf.write(str(SCENES_DIR / scene["filename"]), master_audio, sample_rate, subtype="PCM_16")
    scene["duration_sec"] = round(master_audio.shape[0] / sample_rate, 2)
    _write_scene_package_files(scene)
    _write_json(SCENES_JSON, scenes)
    return {"scene": scene}


@app.get("/api/scenes/{scene_id}/export.zip", tags=["Multi-Character Scene Director"])
def api_export_scene_zip(scene_id: str):
    """
    Download a complete, ready-to-import Game Engine Cutscene Bundle (`.zip`) containing:
      - `master_scene.wav` (24kHz loudness-normalized master audio)
      - `subtitles.srt` (SubRip subtitle file with exact speaker timestamps)
      - `manifest.json` (Unity / Unreal / Godot JSON dialogue cue sheet)
      - `lines/001_<speaker>.wav`, `002_<speaker>.wav`, ... (individual numbered line stems)
    """
    scenes = _read_json(SCENES_JSON, [])
    scene = next((s for s in scenes if s["id"] == scene_id), None)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found.")

    # Ensure .srt and _manifest.json are up to date on disk
    _write_scene_package_files(scene)

    zip_path = SCENES_DIR / f"{scene_id}_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        master_wav = SCENES_DIR / scene["filename"]
        if master_wav.exists():
            zf.write(master_wav, arcname="master_scene.wav")
        srt_file = SCENES_DIR / f"{scene_id}.srt"
        if srt_file.exists():
            zf.write(srt_file, arcname="subtitles.srt")
        manifest_file = SCENES_DIR / f"{scene_id}_manifest.json"
        if manifest_file.exists():
            zf.write(manifest_file, arcname="manifest.json")

        for seg in scene.get("segments", []):
            take_file = TAKES_DIR / Path(seg["take_url"]).name
            if take_file.exists():
                safe_spk = re.sub(r"[^a-zA-Z0-9_-]", "_", seg["speaker"].lower())
                zf.write(take_file, arcname=f"lines/{seg['index']:03d}_{safe_spk}.wav")

    safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", scene.get("title", scene_id).lower()).strip("_") or scene_id
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"{safe_title}_{scene_id}.zip",
    )


@app.get("/api/scenes/{scene_id}/subtitles.srt", tags=["Multi-Character Scene Director"])
def api_get_scene_srt(scene_id: str):
    """Download the `.srt` SubRip subtitle file for a rendered scene."""
    scenes = _read_json(SCENES_JSON, [])
    scene = next((s for s in scenes if s["id"] == scene_id), None)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found.")
    _write_scene_package_files(scene)
    srt_path = SCENES_DIR / f"{scene_id}.srt"
    return FileResponse(path=str(srt_path), media_type="text/plain; charset=utf-8", filename=f"{scene_id}.srt")


@app.get("/api/scenes/{scene_id}/manifest.json", tags=["Multi-Character Scene Director"])
def api_get_scene_manifest(scene_id: str):
    """Download the game-engine dialogue manifest JSON for a rendered scene."""
    scenes = _read_json(SCENES_JSON, [])
    scene = next((s for s in scenes if s["id"] == scene_id), None)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found.")
    _write_scene_package_files(scene)
    manifest_path = SCENES_DIR / f"{scene_id}_manifest.json"
    return _read_json(manifest_path, {})


@app.post("/api/scenes/render/wav", tags=["Multi-Character Scene Director"])
def api_render_scene_wav(req: SceneRenderRequest):
    """
    Render a complete multi-character script and return the stitched master 24kHz `.wav` file directly in the HTTP response body.
    """
    scene_record = _execute_scene_render(req)
    return FileResponse(
        path=scene_record["file_path"],
        media_type="audio/wav",
        filename=scene_record["filename"],
        headers={"X-Scene-Id": scene_record["id"]},
    )


@app.post("/api/scenes/stitch", tags=["Multi-Character Scene Director"])
def api_stitch_existing_takes(req: StitchTakesRequest):
    """Stitch an ordered list of existing `take_ids` into a single combined 24kHz `.wav` scene."""
    takes = _read_json(TAKES_JSON, [])
    take_map = {t["id"]: t for t in takes}

    segments: List[dict] = []
    for idx, tid in enumerate(req.take_ids):
        t = take_map.get(tid)
        if not t:
            raise HTTPException(status_code=404, detail=f"Take '{tid}' not found.")
        wav_path = TAKES_DIR / t["filename"]
        if not wav_path.exists():
            raise HTTPException(status_code=404, detail=f"Audio file for take '{tid}' missing.")

        segments.append({
            "index": idx + 1,
            "speaker": t.get("character_name") or t.get("model_name"),
            "character_id": t.get("character_id"),
            "model_id": t.get("model_id", "qwen3_voicedesign"),
            "action": "stitched_take",
            "text": t.get("text", ""),
            "take_id": tid,
            "take_url": t["url"],
            "duration_sec": t.get("metrics", {}).get("duration_sec", 0.0),
            "start_sec": 0.0,
            "end_sec": 0.0,
            "pause_after_ms": req.pause_ms if idx < len(req.take_ids) - 1 else 0,
        })

    if not segments:
        raise HTTPException(status_code=400, detail="No takes provided to stitch.")

    sr = 24000
    master = _mix_segments_into_master(segments, sample_rate=sr)
    scene_id = f"scene_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    scene_filename = f"{scene_id}.wav"
    scene_filepath = SCENES_DIR / scene_filename
    sf.write(str(scene_filepath), master, sr, subtype="PCM_16")

    scene_record = {
        "id": scene_id,
        "title": req.title.strip() or "Stitched Scene",
        "filename": scene_filename,
        "file_path": str(scene_filepath),
        "url": f"/audio/scenes/{scene_filename}",
        "duration_sec": round(master.shape[0] / sr, 2),
        "elapsed_sec": 0.05,
        "model_swaps": 0,
        "line_count": len(segments),
        "cast": [],
        "segments": segments,
        "created_at": time.strftime("%H:%M:%S"),
    }

    _write_scene_package_files(scene_record)

    scenes = _read_json(SCENES_JSON, [])
    scenes.insert(0, scene_record)
    _write_json(SCENES_JSON, scenes[:50])
    return {"scene": scene_record}


@app.get("/api/scenes", tags=["Multi-Character Scene Director"])
def api_list_scenes():
    """List all stitched multi-character scenes."""
    return {"scenes": _read_json(SCENES_JSON, [])}


@app.delete("/api/scenes/{scene_id}", tags=["Multi-Character Scene Director"])
def api_delete_scene(scene_id: str):
    """Delete a stitched scene and its master WAV file."""
    scenes = _read_json(SCENES_JSON, [])
    target = next((s for s in scenes if s["id"] == scene_id), None)
    if target:
        p = SCENES_DIR / target["filename"]
        if p.exists():
            p.unlink()
    scenes = [s for s in scenes if s["id"] != scene_id]
    _write_json(SCENES_JSON, scenes)
    return {"scenes": scenes}


# ============================================================================
# Casting Desk: Claude-assisted script breakdown, voice casting & saved casts
# ============================================================================

CASTS_JSON = DATA_DIR / "casts.json"

# Cheap model pulls names out of a script; the strong model writes voice descriptions.
CLAUDE_EXTRACT_MODEL = os.environ.get("VOICEFORGE_EXTRACT_MODEL", "haiku")
CLAUDE_IMPROVE_MODEL = os.environ.get("VOICEFORGE_IMPROVE_MODEL", "opus")
CLAUDE_API_MODEL_IDS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}


def _find_claude_cli() -> Optional[str]:
    found = shutil.which("claude")
    if found:
        return found
    for cand in (Path.home() / ".local/bin/claude", Path("/opt/homebrew/bin/claude"), Path("/usr/local/bin/claude")):
        if cand.exists():
            return str(cand)
    return None


def _extract_json_blob(text: str) -> Any:
    """Pull the first JSON object out of a model reply (tolerates ```json fences and chatter)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        return json.loads(t[start:end + 1])
    raise ValueError("Claude did not return JSON.")


def _ask_claude(system: str, prompt: str, schema: dict, model: str, timeout: int = 180, effort: Optional[str] = None) -> dict:
    """
    Ask Claude for a JSON object. Uses the Anthropic API when ANTHROPIC_API_KEY is set,
    otherwise the local `claude` CLI in headless mode (uses Tamair's Claude login, no tools).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        import urllib.request
        body = {
            "model": CLAUDE_API_MODEL_IDS.get(model, model),
            "max_tokens": 4096,
            "system": system + "\n\nReply with a single JSON object matching this schema and nothing else:\n" + json.dumps(schema),
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode(),
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return _extract_json_blob(text)

    cli = _find_claude_cli()
    if not cli:
        raise RuntimeError("Claude is not available: install the `claude` CLI and sign in, or set ANTHROPIC_API_KEY.")

    # Don't let a parent Claude Code session's env leak into the child CLI
    env = {k: v for k, v in os.environ.items() if not (k.startswith("CLAUDE_CODE") or k in ("CLAUDECODE", "ANTHROPIC_BASE_URL"))}
    cmd = [
        cli, "-p", prompt,
        "--model", model,
        "--output-format", "json",
        "--system-prompt", system,
        "--json-schema", json.dumps(schema),
        "--tools", "",
        "--strict-mcp-config",
        "--no-session-persistence",
    ]
    if effort:
        cmd += ["--effort", effort]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(DATA_DIR), env=env)
    try:
        out = json.loads(proc.stdout)
    except Exception:
        raise RuntimeError(f"Claude CLI failed: {(proc.stderr or proc.stdout).strip()[:400]}")
    if out.get("is_error"):
        msg = str(out.get("result", "unknown error"))
        if "authenticate" in msg.lower() or "login" in msg.lower():
            msg += " (run `claude auth login` in a terminal on this Mac)"
        raise RuntimeError(f"Claude CLI error: {msg}")
    if isinstance(out.get("structured_output"), dict):
        return out["structured_output"]
    return _extract_json_blob(str(out.get("result", "")))


EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "suggested_voice_id": {"type": "string"},
                },
                "required": ["name", "description"],
            },
        },
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                    "direction": {"type": "string"},
                    "variant_voice_id": {"type": "string"},
                    "emotion": {"type": "string", "enum": ["neutral", "happy", "excited", "angry", "annoyed", "sad", "scared", "sarcastic", "confused", "pleading", "tired", "whispering", "shouting"]},
                },
                "required": ["speaker", "text"],
            },
        },
    },
    "required": ["characters", "lines"],
}

EXTRACT_SYSTEM = (
    "You break comedy sketches, scripts and loose story notes into spoken dialogue for a text-to-speech studio. "
    "List every character who speaks (merge spelling variants and stage-name casing; keep the most natural form, e.g. 'Dave' not 'DAVE'). "
    "For each one give a short voice description inferred from the script (gender, rough age, accent, personality, how they talk); "
    "say 'unknown' for anything the script doesn't hint at. "
    "If an existing studio voice clearly fits (same name, or an obvious match), put its id in suggested_voice_id, else leave it empty. "
    "Then list the spoken lines in order. Keep the dialogue words exactly; drop stage directions from `text` and put any acting "
    "direction for that line (e.g. 'panicked', 'whispering') in `direction`. "
    "The roster often holds emotion variants of one person (e.g. 'Phoebe', 'Phoebe Panicked', 'Phoebe Pleading'). When a line's mood "
    "clearly matches a variant of the voice you suggested for that speaker, put the variant's id in that line's variant_voice_id; "
    "otherwise leave it empty. Never pick a variant of a different person. "
    "Tag every line with the single closest `emotion` from the allowed list (use 'neutral' when nothing stands out). "
    "If the input is only prose with no dialogue yet, write plausible lines that tell the same beats. "
    "Also suggest a short title."
)


def _roster_brief() -> List[dict]:
    return [
        {"id": c["id"], "name": c["name"], "description": (c.get("description") or "")[:140]}
        for c in _read_json(CHARACTERS_JSON, [])
    ]


def _heuristic_extract(script: str) -> dict:
    """Offline fallback when Claude is unavailable: parse `Name: line` scripts."""
    lines = _parse_script_text_to_lines(script)
    seen: Dict[str, dict] = {}
    for l in lines:
        if l.speaker.isupper():
            l.speaker = l.speaker.title()  # screenplay-style DAVE -> Dave
        key = l.speaker.strip().lower()
        if key not in seen:
            match = _resolve_character(l.speaker)
            seen[key] = {
                "name": l.speaker.strip(),
                "description": l.instruct if _is_voice_description(l.instruct) else "",
                "suggested_voice_id": match["id"] if match else "",
            }
    return {
        "title": "Untitled Scene",
        "characters": list(seen.values()),
        # A short parenthetical like "(panting)" is an acting direction, not a voice description
        "lines": [
            {
                "speaker": l.speaker.strip(),
                "text": l.text,
                "direction": "" if _is_voice_description(l.instruct) else (l.instruct or ""),
                "emotion": _emotion_from_text("" if _is_voice_description(l.instruct) else (l.instruct or ""), l.text),
            }
            for l in lines
        ],
    }


def _is_voice_description(instruct: Optional[str]) -> bool:
    return bool(instruct) and len(instruct.split()) > 3


class CastExtractRequest(BaseModel):
    script: str = Field(..., description="A sketch, script, or a few lines of dialogue / story notes.")
    model: Optional[str] = Field(None, description="Claude model alias for extraction (defaults to a fast cheap model, 'haiku').")
    use_claude: bool = Field(True, description="False skips Claude and uses the built-in `Name: line` parser.")


class CastImproveRequest(BaseModel):
    name: str = Field("Character", description="Character name.")
    description: str = Field("", description="Rough voice description or notes from the user.")
    lines: List[str] = Field(default_factory=list, description="Some of the character's lines from the script, for context.")
    model: Optional[str] = Field(None, description="Claude model alias (defaults to a strong model, 'opus').")


class SavedCastRequest(BaseModel):
    name: str = Field(..., description="Name for this saved cast / scene setup.")
    script: str = Field("", description="Original pasted script.")
    title: str = Field("", description="Scene title.")
    mapping: List[dict] = Field(default_factory=list, description="List of {name, voice_id, script_name?}.")
    lines: List[dict] = Field(default_factory=list, description="List of {speaker, text, direction?}.")
    temperature: Optional[float] = None


class UpdateCharacterRequest(BaseModel):
    name: Optional[str] = Field(None, description="New display name for the locked character voice.")
    description: Optional[str] = Field(None, description="Voice description / notes.")
    ref_text: Optional[str] = Field(None, description="Exact transcript of the reference clip (fixing a bad auto-transcript improves cloning).")
    favorite: Optional[bool] = Field(None, description="Pin this voice to the top of lists.")
    tags: Optional[List[str]] = Field(None, description="Free-form tags, e.g. ['uni sketch', 'irish'].")


@app.get("/cast", response_class=HTMLResponse, include_in_schema=False)
def cast_page():
    return (STATIC_DIR / "cast.html").read_text(encoding="utf-8")


@app.get("/api/cast/claude-status", tags=["Casting Desk"])
def api_cast_claude_status():
    """Report whether the server can reach Claude (API key or a signed-in `claude` CLI)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {"available": True, "via": "api", "detail": "ANTHROPIC_API_KEY"}
    cli = _find_claude_cli()
    if not cli:
        return {"available": False, "via": None, "detail": "claude CLI not found"}
    env = {k: v for k, v in os.environ.items() if not (k.startswith("CLAUDE_CODE") or k in ("CLAUDECODE", "ANTHROPIC_BASE_URL"))}
    try:
        out = subprocess.run([cli, "auth", "status"], capture_output=True, text=True, timeout=20, env=env)
        info = json.loads(out.stdout)
        logged_in = bool(info.get("loggedIn"))
    except Exception:
        logged_in = False
    return {
        "available": logged_in,
        "via": "cli",
        "detail": "signed in" if logged_in else "claude CLI is logged out: run `claude auth login`",
        "extract_model": CLAUDE_EXTRACT_MODEL,
        "improve_model": CLAUDE_IMPROVE_MODEL,
    }


@app.post("/api/cast/extract", tags=["Casting Desk"])
def api_cast_extract(req: CastExtractRequest):
    """
    Break a sketch / script into characters + ordered lines with a fast Claude model,
    suggesting an existing locked voice for each character where one fits.
    Falls back to the built-in `Name: line` parser if Claude is unavailable.
    """
    if not req.script.strip():
        raise HTTPException(status_code=400, detail="Paste a script or some lines first.")
    roster = _roster_brief()
    roster_ids = {c["id"] for c in roster}
    warning = None
    via = "claude"
    if req.use_claude:
        prompt = (
            "Existing studio voices (id, name, description):\n"
            + json.dumps(roster, ensure_ascii=False)
            + "\n\nScript:\n<<<\n" + req.script.strip() + "\n>>>"
        )
        try:
            result = _ask_claude(EXTRACT_SYSTEM, prompt, EXTRACT_SCHEMA, req.model or CLAUDE_EXTRACT_MODEL, effort="low")
        except Exception as e:
            warning = f"{e} Used the built-in parser instead."
            result = _heuristic_extract(req.script)
            via = "parser"
    else:
        result = _heuristic_extract(req.script)
        via = "parser"

    chars = []
    for c in result.get("characters", []):
        name = str(c.get("name", "")).strip()
        if not name:
            continue
        sid = str(c.get("suggested_voice_id") or "").strip()
        if sid not in roster_ids:
            match = _resolve_character(name)
            sid = match["id"] if match else ""
        chars.append({"name": name, "description": str(c.get("description", "")).strip(), "suggested_voice_id": sid})

    # Make sure every speaker in `lines` has a character entry
    known = {c["name"].lower() for c in chars}
    lines = []
    for l in result.get("lines", []):
        spk, txt = str(l.get("speaker", "")).strip(), str(l.get("text", "")).strip()
        if not spk or not txt:
            continue
        if spk.lower() not in known:
            match = _resolve_character(spk)
            chars.append({"name": spk, "description": "", "suggested_voice_id": match["id"] if match else ""})
            known.add(spk.lower())
        variant = str(l.get("variant_voice_id") or "").strip()
        lines.append({
            "speaker": spk,
            "text": txt,
            "direction": str(l.get("direction") or "").strip(),
            "variant_voice_id": variant if variant in roster_ids else "",
            "emotion": str(l.get("emotion") or "") or _emotion_from_text(str(l.get("direction") or ""), txt),
        })

    if not chars:
        raise HTTPException(status_code=422, detail=warning or "No speaking characters found in that script.")
    return {"title": result.get("title") or "Untitled Scene", "characters": chars, "lines": lines, "via": via, "warning": warning}


IMPROVE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "sample_text": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["description", "sample_text"],
}

IMPROVE_SYSTEM = (
    "You are a voice director writing prompts for Qwen3-TTS VoiceDesign, which invents a voice from a plain-English description. "
    "Rewrite the user's rough description into one rich paragraph (60-110 words) covering: persona and attitude, gender and age, accent, "
    "pitch and register, vocal texture, pace and rhythm, and above all emotional range: how the voice moves when excited, upset, sarcastic "
    "or scared, with audible breath, laughter, cracks and emphasis where they fit the character. Keep every concrete fact the user gave; "
    "never change their gender, age or accent. Write it as a description of the person speaking, no markup or tags. "
    "Also write sample_text: 2-3 spoken sentences (25-40 words, plain words, no tags or stage directions) in the character's own voice "
    "that swing between at least two emotions. It becomes the reference clip every later line is cloned from, so it must be lively, "
    "not a flat neutral reading. notes: one short line on what you changed."
)


@app.post("/api/cast/improve-description", tags=["Casting Desk"])
def api_cast_improve_description(req: CastImproveRequest):
    """Optional step: a strong Claude model turns a rough description into an emotionally rich VoiceDesign prompt + lively reference line."""
    if not req.description.strip() and not req.lines:
        raise HTTPException(status_code=400, detail="Give a rough description (or some lines) to improve.")
    prompt = f"Character: {req.name}\nRough description: {req.description.strip() or '(none given)'}\n"
    if req.lines:
        prompt += "Some of their lines in the script:\n" + "\n".join(f"- {t}" for t in req.lines[:12])
    try:
        result = _ask_claude(IMPROVE_SYSTEM, prompt, IMPROVE_SCHEMA, req.model or CLAUDE_IMPROVE_MODEL, timeout=240)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "description": str(result.get("description", "")).strip(),
        "sample_text": str(result.get("sample_text", "")).strip(),
        "notes": str(result.get("notes", "")).strip(),
    }


_WHISPER_MODEL = None


def _transcribe(path: Union[str, Path]) -> str:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import mlx_audio.stt as stt
        _WHISPER_MODEL = stt.load_model("mlx-community/whisper-tiny")
    return _WHISPER_MODEL.generate(str(path)).text.strip()


def _norm_words(text: str) -> List[str]:
    text = re.sub(r"<[^>]+>", " ", text.lower())
    return re.findall(r"[a-z0-9']+", text)


@app.post("/api/scenes/{scene_id}/check", tags=["Casting Desk"])
def api_check_scene(scene_id: str):
    """
    Quality check a rendered scene: transcribe every line with on-device Whisper and compare it with the script,
    and flag lines whose length is implausible for their text (runaways, truncations, long silences).
    Whisper-tiny is rough, so treat a low match as "listen to this one", not proof of a bad take.
    """
    import difflib

    scene = next((s for s in _read_json(SCENES_JSON, []) if s["id"] == scene_id), None)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found.")
    results = []
    for seg in scene.get("segments", []):
        take_path = TAKES_DIR / Path(seg["take_url"]).name
        heard = ""
        try:
            heard = _transcribe(take_path)
        except Exception as e:
            heard = f"(transcription failed: {e})"
        want, got = _norm_words(seg["text"]), _norm_words(heard)
        match = difflib.SequenceMatcher(None, want, got).ratio() if want else 1.0
        n_chars = max(1, len(re.sub(r"<[^>]+>", "", seg["text"]).strip()))
        ms_per_char = seg["duration_sec"] * 1000 / n_chars
        issues = []
        if match < 0.6:
            issues.append("words don't match the script")
        if ms_per_char > 160 and seg["duration_sec"] > 2.5:
            issues.append("much longer than the text (runaway or long pauses)")
        if ms_per_char < 35:
            issues.append("much shorter than the text (words may be cut)")
        results.append({
            "index": seg["index"],
            "speaker": seg["speaker"],
            "text": seg["text"],
            "heard": heard,
            "match": round(match, 2),
            "ms_per_char": round(ms_per_char),
            "issues": issues,
        })
    return {"scene_id": scene_id, "lines": results, "flagged": [r["index"] for r in results if r["issues"]]}


@app.patch("/api/characters/{char_id}", tags=["Character Voice Locker"])
def api_update_character(char_id: str, req: UpdateCharacterRequest):
    """Rename a locked voice or edit its description, reference transcript, tags or favourite flag."""
    chars = _read_json(CHARACTERS_JSON, [])
    target = next((c for c in chars if c["id"] == char_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Character not found.")
    if req.name is not None:
        if not req.name.strip():
            raise HTTPException(status_code=400, detail="Name cannot be empty.")
        target["name"] = req.name.strip()
    if req.description is not None:
        target["description"] = req.description.strip()
    if req.ref_text is not None:
        if not req.ref_text.strip():
            raise HTTPException(status_code=400, detail="Reference transcript cannot be empty.")
        target["ref_text"] = req.ref_text.strip()
    if req.favorite is not None:
        target["favorite"] = req.favorite
    if req.tags is not None:
        target["tags"] = sorted({t.strip() for t in req.tags if t.strip()}, key=str.lower)
    _write_json(CHARACTERS_JSON, chars)
    return {"character": target, "characters": chars}


@app.get("/api/voices", tags=["Character Voice Locker"])
def api_list_voices():
    """Locked voices plus reference clip length and how many saved scene lines use each one (for the voice manager)."""
    chars = _read_json(CHARACTERS_JSON, [])
    usage: Dict[str, int] = {}
    for scene in _read_json(SCENES_JSON, []):
        for seg in scene.get("segments", []):
            cid = seg.get("character_id")
            if cid:
                usage[cid] = usage.get(cid, 0) + 1
    voices = []
    for c in chars:
        path = CHARACTERS_DIR / c.get("audio_file", "")
        try:
            info = sf.info(str(path))
            duration = round(info.frames / info.samplerate, 2)
        except Exception:
            duration = None
        voices.append({**c, "duration_sec": duration, "scene_lines": usage.get(c["id"], 0), "missing_audio": not path.exists()})
    return {"voices": voices}


@app.post("/api/characters/{char_id}/retranscribe", tags=["Character Voice Locker"])
def api_retranscribe_character(char_id: str):
    """Re-run on-device Whisper on a voice's reference clip and return the transcript (not saved until PATCHed)."""
    char = next((c for c in _read_json(CHARACTERS_JSON, []) if c["id"] == char_id), None)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found.")
    try:
        return {"ref_text": _transcribe(CHARACTERS_DIR / char["audio_file"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@app.get("/voices", response_class=HTMLResponse, include_in_schema=False)
def voices_page():
    return (STATIC_DIR / "voices.html").read_text(encoding="utf-8")


@app.get("/api/casts", tags=["Casting Desk"])
def api_list_casts():
    """List saved cast setups (script + name-to-voice mapping)."""
    return {"casts": _read_json(CASTS_JSON, [])}


@app.post("/api/casts", tags=["Casting Desk"])
def api_save_cast(req: SavedCastRequest):
    """Save (or overwrite by name) a cast setup so a sketch can be re-rendered later with the same voices."""
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Cast name is required.")
    casts = _read_json(CASTS_JSON, [])
    existing = next((c for c in casts if c["name"].lower() == req.name.strip().lower()), None)
    record = req.model_dump()
    record["name"] = req.name.strip()
    record["id"] = existing["id"] if existing else f"cast_{int(time.time())}_{uuid.uuid4().hex[:5]}"
    record["updated_at"] = time.strftime("%Y-%m-%d %H:%M")
    casts = [c for c in casts if c["id"] != record["id"]]
    casts.insert(0, record)
    _write_json(CASTS_JSON, casts)
    return {"cast": record, "casts": casts}


@app.delete("/api/casts/{cast_id}", tags=["Casting Desk"])
def api_delete_cast(cast_id: str):
    casts = [c for c in _read_json(CASTS_JSON, []) if c["id"] != cast_id]
    _write_json(CASTS_JSON, casts)
    return {"casts": casts}


# ============================================================================
# Clone Studio: character profiles holding one cloned clip per emotion
# ============================================================================

PROFILES_JSON = DATA_DIR / "profiles.json"

# Each emotion: how to perform it, a few things to read (8-12s each), and which emotions can stand in for it.
EMOTIONS: List[dict] = [
    {"id": "neutral", "label": "Neutral", "icon": "🙂", "fallback": [],
     "how": "Relaxed and conversational, like telling a mate about your day. Not a flat reading voice: let it rise and fall naturally.",
     "scripts": [
         "So I got to the station about ten minutes early, which never happens, and of course the train was already sitting there waiting for me.",
         "Honestly, the best bit of the whole weekend was just sitting outside with a coffee and doing absolutely nothing for an hour.",
         "Right, so the plan is we meet at the corner around seven, grab some food, and then see how everyone's feeling after that.",
     ]},
    {"id": "happy", "label": "Happy", "icon": "😄", "fallback": ["excited", "neutral"],
     "how": "Smile while you speak; it really changes the sound. Warm, bright, a little laugh in the voice is great.",
     "scripts": [
         "No way, you actually remembered? That's so nice of you. Honestly, you've made my whole week, I didn't think anyone would notice.",
         "Oh, this is brilliant. Look at it! It's exactly what I wanted. I can't believe you found the same one.",
         "Ha, okay, that's genuinely the funniest thing I've heard all day. Tell it again, tell it again, I need to hear it properly.",
     ]},
    {"id": "excited", "label": "Excited", "icon": "🤩", "fallback": ["happy", "neutral"],
     "how": "Fast, high energy, words tumbling out. Lean towards the mic a bit less so the louder bits don't clip.",
     "scripts": [
         "Wait, wait, wait, are you serious? We got the tickets? Front row? Oh my god, okay, I need to sit down, this is actually happening!",
         "Guess what, guess what! They said yes! I start on Monday! I literally screamed in the car park, people were staring at me!",
         "Okay, okay, listen to this idea, because it's genius. We take the van, we drive to the coast, and we don't come back until Sunday!",
     ]},
    {"id": "angry", "label": "Angry", "icon": "😠", "fallback": ["annoyed", "shouting", "neutral"],
     "how": "Controlled fury: tight, clipped and forceful, but not screaming. Hit the consonants hard.",
     "scripts": [
         "No. Don't you dare tell me to calm down. I asked you three times, three times, and you looked me in the eye and said it was done.",
         "You know what, I'm done being polite about it. You took it without asking, you broke it, and now you're blaming me?",
         "Every single time. Every single time I trust you with something, this happens, and I'm the idiot for being surprised.",
     ]},
    {"id": "annoyed", "label": "Annoyed", "icon": "😤", "fallback": ["angry", "sarcastic", "neutral"],
     "how": "Exasperated and fed up, lots of sighing energy. Not shouting, just completely over it.",
     "scripts": [
         "Oh, brilliant. Fantastic. So the one day I actually need the printer to work, it decides to have a little breakdown. Of course it does.",
         "Can you just, can you not tap the table like that? I've asked you nicely about four times now and I'm losing my mind.",
         "Right, so we're doing this again. Same argument, same ending. Wake me up when you've worked out where you left the keys.",
     ]},
    {"id": "sad", "label": "Sad", "icon": "😢", "fallback": ["tired", "neutral"],
     "how": "Slower and softer, voice dropping at the end of sentences. Let it wobble or catch; don't push it.",
     "scripts": [
         "I just thought it would be different this time, you know? I really thought he'd come. I kept looking at the door all night.",
         "It's fine. Honestly, it's fine. I'm not upset. I just, I wish someone had told me before I found out like that.",
         "We used to sit right here every Sunday. It feels really strange coming back now that it's just me.",
     ]},
    {"id": "scared", "label": "Scared", "icon": "😨", "fallback": ["pleading", "excited", "neutral"],
     "how": "Breathy, fast and shaky, a bit hushed as if someone might hear. Short panicked breaths between phrases are good.",
     "scripts": [
         "Did you hear that? No, listen, there's something outside. Something's moving by the window. Please tell me you locked the back door.",
         "Okay, okay, don't panic. The lights just went off, that's all. It's fine. It's fine. Why is the door open? I closed that door.",
         "Stop the car. Stop the car! There's someone standing in the road, right there, can you not see them?",
     ]},
    {"id": "sarcastic", "label": "Sarcastic", "icon": "🙄", "fallback": ["annoyed", "neutral"],
     "how": "Dry and deadpan with exaggerated emphasis on the 'nice' words. Slight drawl.",
     "scripts": [
         "Oh wow, what an amazing surprise, you're late again. I was so worried you'd suddenly turn up on time and ruin the tradition.",
         "No, no, please, take the last slice. You clearly need it more than me. I only paid for the whole pizza.",
         "Yeah, great plan. Really solid. Nothing could possibly go wrong with driving there without a map at midnight.",
     ]},
    {"id": "confused", "label": "Confused", "icon": "🤔", "fallback": ["neutral"],
     "how": "Puzzled and hesitant, questions going up at the end, little pauses while you work it out.",
     "scripts": [
         "Sorry, what? Hang on. So you're saying the meeting was yesterday? But the email said Thursday. Today's Thursday. Isn't it?",
         "Wait, I don't get it. Why would you put the milk in the cupboard? And where's the cereal gone? Is it in the fridge?",
         "Hold on, who's Dave? Is that the same Dave from before, or a different Dave? Because I'm really lost now.",
     ]},
    {"id": "pleading", "label": "Pleading", "icon": "🥺", "fallback": ["sad", "scared", "neutral"],
     "how": "Desperate and earnest, voice pushing upward. Really mean it, as if this is your last chance.",
     "scripts": [
         "Please, just give me five minutes. That's all I'm asking. Five minutes and I'll explain everything, I promise, please.",
         "Come on, don't do this. I'm begging you. I'll do anything, I'll do the dishes for a month, just please don't tell Mum.",
         "I know I messed up. I know. But please, please don't leave like this. Just stay and talk to me.",
     ]},
    {"id": "tired", "label": "Tired", "icon": "😮‍💨", "fallback": ["sad", "neutral"],
     "how": "Low energy and breathy, slow, yawning or out of breath. Let the words drag.",
     "scripts": [
         "I've been up since five. I haven't eaten. I just want to lie down on the floor right here and not move until tomorrow.",
         "Give me a second. Those stairs. Why are there so many stairs? Who designed this building? Okay. I'm fine. I'm fine.",
         "Mm, yeah, sounds good. Wake me up when we get there. Or don't. Honestly I could sleep for a week.",
     ]},
    {"id": "whispering", "label": "Whispering", "icon": "🤫", "fallback": ["scared", "neutral"],
     "how": "A real hushed stage whisper with some voice in it, not pure air. Get a little closer to the mic and turn your input gain up.",
     "scripts": [
         "Shh, keep your voice down. She's in the next room. If she finds out we've been in here, we are both in so much trouble.",
         "Okay, on three we run for the door. Don't look back, and whatever you do, don't step on the creaky floorboard.",
         "Psst. Are you awake? I can't sleep. I think I heard something downstairs. Do you want to go and check?",
     ]},
    {"id": "shouting", "label": "Shouting", "icon": "📢", "fallback": ["angry", "excited", "neutral"],
     "how": "Projecting across a room. Step back from the mic or turn the gain down so it doesn't distort; check the level meter stays out of the red.",
     "scripts": [
         "Oi! Get off the grass! Can you not read the sign? It's right there in massive letters! Off! Now!",
         "Everyone out! The kitchen's on fire! Leave your bags, just get out of the building, right now, go, go, go!",
         "I'm over here! Over here! Can you hear me? Throw the rope! No, the other way! The other way!",
     ]},
]
EMOTION_IDS = [e["id"] for e in EMOTIONS]

# Keywords in a line's acting direction that point at an emotion slot
EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "shouting": ["shout", "yell", "scream", "bellow", "roar", "caps"],
    "whispering": ["whisper", "hushed", "under breath", "murmur", "sotto voce"],
    "angry": ["angry", "furious", "rage", "livid", "losing it", "seething", "fuming"],
    "annoyed": ["annoyed", "irritated", "exasperat", "frustrat", "fed up", "impatient", "grumpy"],
    "scared": ["scared", "afraid", "panic", "terrified", "nervous", "anxious", "fear", "frightened", "shaky"],
    "sad": ["sad", "upset", "crying", "tearful", "defeated", "hurt", "heartbroken", "gloomy", "devastated"],
    "pleading": ["plead", "beg", "desperate", "imploring"],
    "sarcastic": ["sarcas", "deadpan", "dry", "mocking", "ironic", "snide"],
    "confused": ["confus", "puzzled", "baffled", "bewildered", "unsure", "lost"],
    "excited": ["excited", "thrilled", "ecstatic", "enthusias", "hyped", "elated"],
    "happy": ["happy", "cheerful", "warm", "delighted", "laugh", "amused", "joyful", "pleased", "smiling"],
    "tired": ["tired", "exhausted", "weary", "sleepy", "winded", "out of breath", "panting", "yawn"],
}


def _emotion_from_text(direction: str, text: str = "") -> str:
    d = (direction or "").lower()
    for emo, words in EMOTION_KEYWORDS.items():
        if any(w in d for w in words):
            return emo
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7 and len(letters) > 6:
        return "shouting"
    return "neutral"


def _pick_profile_clip(profile: dict, emotion: str) -> Optional[dict]:
    """Best clip for an emotion: exact slot, then that emotion's stand-ins, then neutral, then anything recorded."""
    clips = {c["emotion"]: c for c in profile.get("clips", [])}
    if not clips:
        return None
    meta = next((e for e in EMOTIONS if e["id"] == emotion), None)
    for emo in [emotion] + (meta["fallback"] if meta else []) + ["neutral"]:
        if emo in clips:
            return clips[emo]
    return next(iter(clips.values()))


def _analyze_recording(path: Union[str, Path]) -> dict:
    """Level, noise and clipping check for a reference recording, with plain-English tips."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = audio.shape[0] / sr
    eps = 1e-9
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    hop = max(1, int(sr * 0.02))
    n = audio.shape[0] // hop
    frames = audio[: n * hop].reshape(n, hop) if n else np.zeros((1, hop), dtype=np.float32)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    speech_db = 20 * np.log10(float(np.percentile(rms, 90)) + eps)
    noise_db = 20 * np.log10(float(np.percentile(rms, 10)) + eps)
    clipped = float(np.mean(np.abs(audio) > 0.985)) if audio.size else 0.0
    tips = []
    if clipped > 0.0005 or peak > 0.995:
        tips.append("It clipped (too loud). Back off the mic a little or lower the input gain; distortion gets cloned too.")
    elif speech_db < -32:
        tips.append("Quite quiet. Move a bit closer (about a hand-span away) or raise the input gain.")
    if noise_db > -50:
        tips.append("Noticeable background noise between words. Turn off fans or AC, close the window, record in a room with soft furnishings.")
    if speech_db - noise_db < 25:
        tips.append("Your voice isn't far above the background noise, so the clone may pick up hiss or room sound.")
    _, ramp = _repair_gain_ramp(audio, sr)
    if ramp:
        tips.append(
            f"The first {ramp['ramp_sec']}s came in about {round(ramp['lift_db'])} dB quieter than the rest (the mic's gain was "
            "still settling). Browser takes are evened out automatically; keeping the mic switched on between takes avoids it."
        )
    if duration < 5:
        tips.append("Short clip. 8-12 seconds gives the clone more to learn from.")
    elif duration > 20:
        tips.append("Long clip. It was trimmed to the most expressive 6-12 seconds automatically.")
    score = 100
    score -= 35 if (clipped > 0.0005 or peak > 0.995) else 0
    score -= 20 if speech_db < -32 else 0
    score -= 20 if noise_db > -50 else 0
    score -= 15 if speech_db - noise_db < 25 else 0
    score -= 10 if duration < 5 else 0
    return {
        "duration_sec": round(duration, 2),
        "peak_dbfs": round(20 * np.log10(peak + eps), 1),
        "speech_dbfs": round(speech_db, 1),
        "noise_floor_dbfs": round(noise_db, 1),
        "clipping_pct": round(clipped * 100, 3),
        "gain_ramp": ramp,
        "score": max(0, score),
        "tips": tips,
    }


def _profiles_with_clips() -> List[dict]:
    chars = {c["id"]: c for c in _read_json(CHARACTERS_JSON, [])}
    out = []
    for p in _read_json(PROFILES_JSON, []):
        clips = []
        for clip in p.get("clips", []):
            ch = chars.get(clip["character_id"])
            if ch:  # drop clips whose voice was deleted in the library
                clips.append({**clip, "name": ch["name"], "url": ch["url"], "ref_text": ch.get("ref_text", "")})
        out.append({**p, "clips": clips})
    return out


def _save_profile(profile: dict) -> None:
    profiles = [p for p in _read_json(PROFILES_JSON, []) if p["id"] != profile["id"]]
    profiles.insert(0, profile)
    _write_json(PROFILES_JSON, profiles)


def _get_profile(profile_id: str) -> dict:
    p = next((p for p in _read_json(PROFILES_JSON, []) if p["id"] == profile_id), None)
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return p


class ProfileRequest(BaseModel):
    name: str = Field(..., description="Person / character name, e.g. 'Tamair'.")
    description: str = Field("", description="Optional notes about the voice.")


class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class LinkClipRequest(BaseModel):
    emotion: str = Field(..., description="Emotion slot id, e.g. 'angry'.")
    character_id: str = Field(..., description="Existing locked voice to use for that emotion.")


class PromptRequest(BaseModel):
    emotion: str = Field(..., description="Emotion slot id.")
    context: str = Field("", description="Optional: who the character is, so the lines fit them.")


@app.get("/clone", response_class=HTMLResponse, include_in_schema=False)
def clone_page():
    return (STATIC_DIR / "clone.html").read_text(encoding="utf-8")


@app.get("/api/profiles/emotions", tags=["Clone Studio"])
def api_profile_emotions():
    """Emotion slots with acting directions and suggested lines to read, plus the direction keywords used to auto-pick clips."""
    return {"emotions": EMOTIONS, "keywords": EMOTION_KEYWORDS}


@app.get("/api/profiles", tags=["Clone Studio"])
def api_list_profiles():
    """Character profiles: one person with a cloned reference clip per emotion."""
    return {"profiles": _profiles_with_clips()}


@app.post("/api/profiles", tags=["Clone Studio"])
def api_create_profile(req: ProfileRequest):
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required.")
    profile = {
        "id": f"prof_{int(time.time())}_{uuid.uuid4().hex[:5]}",
        "name": req.name.strip(),
        "description": req.description.strip(),
        "clips": [],
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    _save_profile(profile)
    return {"profile": profile, "profiles": _profiles_with_clips()}


@app.patch("/api/profiles/{profile_id}", tags=["Clone Studio"])
def api_update_profile(profile_id: str, req: ProfileUpdateRequest):
    p = _get_profile(profile_id)
    if req.name is not None and req.name.strip():
        p["name"] = req.name.strip()
    if req.description is not None:
        p["description"] = req.description.strip()
    _save_profile(p)
    return {"profiles": _profiles_with_clips()}


@app.delete("/api/profiles/{profile_id}", tags=["Clone Studio"])
def api_delete_profile(profile_id: str, delete_voices: bool = Query(False, description="Also delete the clip voices from the library.")):
    p = _get_profile(profile_id)
    if delete_voices:
        # Only voices recorded for this profile; voices linked in from the library are left alone
        owned = {c["id"] for c in _read_json(CHARACTERS_JSON, []) if c.get("profile_id") == profile_id}
        for clip in p.get("clips", []):
            if clip["character_id"] in owned:
                api_delete_character(clip["character_id"])
    _write_json(PROFILES_JSON, [x for x in _read_json(PROFILES_JSON, []) if x["id"] != profile_id])
    return {"profiles": _profiles_with_clips()}


@app.post("/api/profiles/{profile_id}/clips", tags=["Clone Studio"])
async def api_add_profile_clip(
    profile_id: str,
    emotion: str = Form(..., description="Emotion slot id."),
    script: str = Form("", description="What the user was asked to read; used as the transcript when Whisper agrees with it."),
    source: str = Form("upload", description="'browser' for takes recorded on the Clone Studio page (their quiet-start mic ramp is repaired automatically)."),
    file: UploadFile = File(..., description="Recording (wav, mp3, m4a, mp4, webm…)."),
):
    """Record or upload one emotion for a profile. Cleans it into a 24kHz reference, transcribes it, checks levels and noise."""
    import difflib

    if emotion not in EMOTION_IDS:
        raise HTTPException(status_code=400, detail=f"Unknown emotion '{emotion}'.")
    profile = _get_profile(profile_id)
    label = next(e["label"] for e in EMOTIONS if e["id"] == emotion)

    ext = Path(file.filename or "rec.webm").suffix or ".webm"
    temp_path = DATA_DIR / f"upload_{int(time.time())}_{uuid.uuid4().hex[:5]}{ext}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        quality = None
        try:
            # Analyse the untrimmed take so tips reflect the whole recording
            wav_probe = temp_path.with_suffix(".probe.wav")
            subprocess.run(["ffmpeg", "-y", "-i", str(temp_path), "-ac", "1", "-ar", "24000", str(wav_probe)],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            quality = _analyze_recording(wav_probe)
            wav_probe.unlink(missing_ok=True)
        except Exception:
            quality = None
        char = _ingest_audio_recording(
            temp_path,
            name=f"{profile['name']} · {label}",
            description=f"{profile['name']}, {label.lower()} delivery. {profile.get('description', '')}".strip(),
            repair_gain_ramp=(source == "browser"),
        )
    finally:
        temp_path.unlink(missing_ok=True)

    # Prefer the clean script as the transcript when Whisper heard roughly the same words
    heard = char.get("ref_text", "")
    match = difflib.SequenceMatcher(None, _norm_words(script), _norm_words(heard)).ratio() if script.strip() else 0.0
    chars = _read_json(CHARACTERS_JSON, [])
    stored = next(c for c in chars if c["id"] == char["id"])
    if script.strip() and match >= 0.75:
        stored["ref_text"] = script.strip()
    stored["tags"] = sorted({profile["name"], label.lower(), "profile"}, key=str.lower)
    stored["profile_id"] = profile_id
    _write_json(CHARACTERS_JSON, chars)

    profile = _get_profile(profile_id)
    profile["clips"] = [c for c in profile.get("clips", []) if c["emotion"] != emotion] + [
        {"emotion": emotion, "character_id": char["id"], "quality": quality, "added_at": time.strftime("%Y-%m-%d %H:%M")}
    ]
    _save_profile(profile)
    return {
        "character": stored,
        "heard": heard,
        "script_match": round(match, 2),
        "used_script_as_transcript": bool(script.strip() and match >= 0.75),
        "quality": quality,
        "profiles": _profiles_with_clips(),
    }


@app.post("/api/profiles/{profile_id}/repair", tags=["Clone Studio"])
def api_repair_profile_clips(profile_id: str):
    """Even out quiet starts (mic gain still settling) on every clip recorded for this profile. Originals are kept in characters/originals/."""
    profile = _get_profile(profile_id)
    chars = _read_json(CHARACTERS_JSON, [])
    by_id = {c["id"]: c for c in chars}
    results = {}
    for clip in profile.get("clips", []):
        ch = by_id.get(clip["character_id"])
        if not ch:
            continue
        info = _repair_character_audio(ch)
        results[clip["emotion"]] = info
        if info:
            ch["gain_ramp_fixed"] = info
            if clip.get("quality"):
                clip["quality"]["gain_ramp_fixed"] = info
    _write_json(CHARACTERS_JSON, chars)
    _save_profile(profile)
    return {"repaired": results, "profiles": _profiles_with_clips()}


@app.post("/api/profiles/{profile_id}/clips/link", tags=["Clone Studio"])
def api_link_profile_clip(profile_id: str, req: LinkClipRequest):
    """Use an existing library voice for one of the profile's emotion slots."""
    if req.emotion not in EMOTION_IDS:
        raise HTTPException(status_code=400, detail=f"Unknown emotion '{req.emotion}'.")
    if not any(c["id"] == req.character_id for c in _read_json(CHARACTERS_JSON, [])):
        raise HTTPException(status_code=404, detail="Voice not found.")
    profile = _get_profile(profile_id)
    profile["clips"] = [c for c in profile.get("clips", []) if c["emotion"] != req.emotion] + [
        {"emotion": req.emotion, "character_id": req.character_id, "quality": None, "added_at": time.strftime("%Y-%m-%d %H:%M")}
    ]
    _save_profile(profile)
    return {"profiles": _profiles_with_clips()}


@app.delete("/api/profiles/{profile_id}/clips/{emotion}", tags=["Clone Studio"])
def api_unlink_profile_clip(profile_id: str, emotion: str):
    """Remove an emotion slot from the profile (the voice stays in the library)."""
    profile = _get_profile(profile_id)
    profile["clips"] = [c for c in profile.get("clips", []) if c["emotion"] != emotion]
    _save_profile(profile)
    return {"profiles": _profiles_with_clips()}


@app.get("/api/profiles/{profile_id}/pick", tags=["Clone Studio"])
def api_pick_profile_clip(profile_id: str, emotion: str = Query("neutral"), direction: str = Query(""), text: str = Query("")):
    """Which clip a line would use: explicit emotion, else guessed from its acting direction / text."""
    profile = next((p for p in _profiles_with_clips() if p["id"] == profile_id), None)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")
    emo = emotion if emotion in EMOTION_IDS and emotion != "neutral" else _emotion_from_text(direction, text)
    return {"emotion": emo, "clip": _pick_profile_clip(profile, emo)}


PROMPT_SCHEMA = {
    "type": "object",
    "properties": {"script": {"type": "string"}, "how": {"type": "string"}},
    "required": ["script", "how"],
}


@app.post("/api/profiles/prompts", tags=["Clone Studio"])
def api_generate_recording_prompt(req: PromptRequest):
    """Ask Claude for a fresh line to read for an emotion (optionally in a character's voice), with a short acting note."""
    meta = next((e for e in EMOTIONS if e["id"] == req.emotion), None)
    if not meta:
        raise HTTPException(status_code=400, detail=f"Unknown emotion '{req.emotion}'.")
    system = (
        "You write short lines for someone to read aloud into a microphone to record a voice-clone reference clip. "
        "Write 2-3 natural spoken sentences (25-35 words, about 10 seconds), plain words only, no stage directions, "
        "everyday British English, that make it easy to perform the requested emotion convincingly. "
        "`how` is one sentence of acting direction for the reader."
    )
    prompt = f"Emotion: {meta['label']} ({meta['how']})\n"
    if req.context.strip():
        prompt += f"Character / context: {req.context.strip()}\n"
    try:
        return _ask_claude(system, prompt, PROMPT_SCHEMA, CLAUDE_EXTRACT_MODEL, timeout=120, effort="low")
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


app.mount("/audio/scenes", StaticFiles(directory=str(SCENES_DIR)), name="scenes")
app.mount("/audio/takes", StaticFiles(directory=str(TAKES_DIR)), name="takes")
app.mount("/audio/characters", StaticFiles(directory=str(CHARACTERS_DIR)), name="characters")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    index_file = STATIC_DIR / "index.html"
    return index_file.read_text(encoding="utf-8")
