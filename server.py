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
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import VoiceForgeEngine, MODEL_REGISTRY, MAYA1_EMOTION_TAGS

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
        1.15,
        ge=0.5,
        le=2.0,
        description="Pitch-preserving speaking speed multiplier (1.0 = normal, 1.15 = snappy default, 1.35 = fast).",
    )
    trim_silence: bool = Field(
        True,
        description="Automatically tighten awkward dead-air pauses (>120ms) between phrases.",
    )
    temperature: Optional[float] = Field(
        None,
        description="Sampling temperature (defaults to 0.80 for Qwen3, 0.45 for Maya-1).",
    )
    top_p: Optional[float] = Field(
        None,
        description="Nucleus sampling top_p (defaults to 0.95 for Qwen3, 0.90 for Maya-1).",
    )
    repetition_penalty: Optional[float] = Field(
        None,
        description="Repetition penalty (defaults to 1.08 for Qwen3, 1.15 for Maya-1).",
    )
    character_id: Optional[str] = Field(
        None,
        description="ID or exact name of a locked character from `/api/characters` when using `model_id='qwen3_clone'`.",
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
    speed: float = Field(1.15, ge=0.5, le=2.0, description="Speaking speed multiplier.")
    trim_silence: bool = Field(True, description="Tighten pauses in the generated golden sample.")


classSaveCharacterFromTakeRequest = BaseModel


class SaveCharacterFromTakeRequest(BaseModel):
    take_id: str = Field(..., description="ID of an existing generated take from `/api/takes`.")
    name: str = Field(..., description="Character name to save in the Character Voice Locker.")
    description: str = Field("", description="Optional notes or voice description.")


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
    speed: float = Field(1.15, ge=0.5, le=2.0, description="Speaking speed multiplier (0.5 to 2.0).")


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
    pause_after_ms: Optional[int] = Field(
        None,
        ge=-2000,
        le=5000,
        description="Gap in milliseconds after this line before the next character speaks. Positive values add silence; negative values (e.g. -200) overlap/interrupt the next speaker!",
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


class SceneRenderRequest(BaseModel):
    """
    Multi-character Scene & Script Renderer for LLMs and Game Engines.
    Provide either:
      - `lines`: Structured list of `{speaker, text, instruct?, pause_after_ms?}` objects, OR
      - `script_text`: Plain-text script where each line is `Speaker Name (optional voice description): Dialogue text`
    Automatically reuses existing locked characters by name, or creates + locks brand-new characters on the fly,
    synthesizes every segment, and stitches them into a single master 24kHz `.wav` scene with natural turn-taking pauses!
    """
    title: str = Field("Game Cutscene", description="Title of the scene.")
    cast: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional dictionary mapping Character Name -> Voice Description (`instruct`) for new characters in the scene.",
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
        1.15,
        ge=0.5,
        le=2.0,
        description="Default speaking speed multiplier across the scene.",
    )
    trim_silence: bool = Field(
        True,
        description="Tighten internal pauses within each segment before stitching.",
    )
    auto_lock_new_characters: bool = Field(
        True,
        description="Automatically lock any newly designed character into the persistent Character Voice Locker so their voice stays identical across the rest of the scene and future scenes.",
    )


class BatchSceneRenderRequest(BaseModel):
    """Render an entire batch of game skits / cutscenes in a single low-friction pipeline call."""
    scenes: List[SceneRenderRequest] = Field(..., description="List of multi-character scenes/skits to render sequentially.")


class StitchTakesRequest(BaseModel):
    """Stitch an arbitrary list of existing `take_ids` into a single combined `.wav` scene."""
    title: str = Field("Stitched Scene", description="Title for the stitched scene.")
    take_ids: List[str] = Field(..., description="Ordered list of `take_id` strings to stitch together.")
    pause_ms: int = Field(280, ge=0, le=5000, description="Silence gap in milliseconds between takes.")


# ============================================================================
# Core Discovery & Status Endpoints
# ============================================================================

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

    if req.model_id == "qwen3_clone":
        if not req.character_id:
            raise HTTPException(
                status_code=400,
                detail="Please provide `character_id` (or character name) from `/api/characters` when using `qwen3_clone`.",
            )
        char = _resolve_character(req.character_id)
        if not char:
            raise HTTPException(status_code=404, detail=f"Character '{req.character_id}' not found in Character Locker.")
        ref_audio_path = str(CHARACTERS_DIR / char["audio_file"])
        ref_text = char["ref_text"]
        character_name = char["name"]
        resolved_char_id = char["id"]

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
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
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
            speed=req.speed,
            trim_silence=req.trim_silence,
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


@app.post("/api/characters/upload", tags=["Character Voice Locker"])
async def api_upload_character(
    name: str = Form(..., description="Character name"),
    ref_text: str = Form(..., description="Exact transcript spoken in the reference audio"),
    description: str = Form("", description="Optional character notes"),
    file: UploadFile = File(..., description="Reference WAV/MP3 audio clip (3s-15s)"),
):
    """Upload an external reference audio clip + transcript to create a locked character voice."""
    if not name.strip() or not ref_text.strip():
        raise HTTPException(status_code=400, detail="Character name and exact reference transcript are required.")

    char_id = f"char_{int(time.time())}_{uuid.uuid4().hex[:5]}"
    ext = Path(file.filename or "ref.wav").suffix or ".wav"
    dest_filename = f"{char_id}{ext}"
    dest_path = CHARACTERS_DIR / dest_filename

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    char_record = {
        "id": char_id,
        "name": name.strip(),
        "description": description.strip() or "Uploaded Reference Audio",
        "ref_text": ref_text.strip(),
        "audio_file": dest_filename,
        "file_path": str(dest_path),
        "url": f"/audio/characters/{dest_filename}",
        "source_model": "Custom Upload",
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }

    chars = _read_json(CHARACTERS_JSON, [])
    chars.insert(0, char_record)
    _write_json(CHARACTERS_JSON, chars)
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
    Supports low-friction formats like:
      - `Torvin (gruff 60s Scottish dwarf blacksmith): Keep your coin, traveler.`
      - `Nyx [maya1, 1.25x]: Oh, brilliant plan! <laugh> Let's walk right in.`
      - `Guard (nervous young watchman) [fast, overlap 180ms]: Wait—don't touch that!`
      - `[Pause: 600ms]` or `[Overlap: 200ms]` (attaches pause/overlap to previous line)
    """
    parsed: List[SceneLineItem] = []
    line_pattern = re.compile(r"^\s*([^\(:\[]+?)(?:\s*\(([^\)]+)\))?(?:\s*\[([^\]]+)\])?\s*:\s*(.+)$")
    pause_pattern = re.compile(r"^\s*\[?\s*(pause|overlap)\s*[:=]?\s*(-?\d+)\s*(?:ms)?\s*\]?\s*$", re.IGNORECASE)

    for raw_line in script_text.splitlines():
        s = raw_line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue

        pm = pause_pattern.match(s)
        if pm:
            if parsed:
                kind = pm.group(1).lower()
                val = abs(int(pm.group(2)))
                parsed[-1].pause_after_ms = -val if kind == "overlap" else val
            continue

        m = line_pattern.match(s)
        if m:
            speaker = m.group(1).strip()
            instruct = (m.group(2) or "").strip() or None
            directives = (m.group(3) or "").strip().lower()
            dialogue = m.group(4).strip()

            model_id = None
            speed = None
            pause_after_ms = None

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
                    speed = 1.30
                elif "slow" in directives:
                    speed = 0.92

                ov_m = re.search(r"overlap\s*[:=]?\s*(\d+)", directives)
                if ov_m:
                    pause_after_ms = -abs(int(ov_m.group(1)))
                ps_m = re.search(r"pause\s*[:=]?\s*(-?\d+)", directives)
                if ps_m:
                    pause_after_ms = int(ps_m.group(1))

            if dialogue:
                parsed.append(
                    SceneLineItem(
                        speaker=speaker,
                        text=dialogue,
                        instruct=instruct,
                        model_id=model_id,
                        speed=speed,
                        pause_after_ms=pause_after_ms,
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


def _mix_segments_into_master(segment_records: List[dict], sample_rate: int = 24000) -> np.ndarray:
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
    cast_descriptions: Dict[str, str] = {k.strip().lower(): v.strip() for k, v in (req.cast or {}).items() if v.strip()}
    for item in lines:
        if item.instruct and item.speaker.strip().lower() not in cast_descriptions:
            cast_descriptions[item.speaker.strip().lower()] = item.instruct.strip()

    pre_rendered_takes: Dict[int, tuple] = {}
    resolved_cast_map: Dict[str, dict] = {}
    cast_summary: Dict[str, dict] = {}
    model_swaps = 0
    last_used_model = engine.active_model_id

    # -------------------------------------------------------------------------
    # PASS 1: Discover brand-new characters, batch by design engine, and lock!
    # -------------------------------------------------------------------------
    new_speaker_first_line: Dict[str, int] = {}
    for idx, line in enumerate(lines):
        speaker = line.speaker.strip()
        s_key = speaker.lower()
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
        if design_engine != last_used_model:
            model_swaps += 1
            last_used_model = design_engine

        take = _run_synthesis_and_save(
            SynthesizeRequest(
                model_id=design_engine,
                text=line.text.strip(),
                instruct=instruct,
                speed=line_speed,
                trim_silence=req.trim_silence,
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
        if l.model_id == "maya1":
            return "maya1"
        if l.model_id == "qwen3_voicedesign":
            return "qwen3_voicedesign"
        if has_locked:
            return "qwen3_clone"
        return "maya1" if ("<" in l.text and ">" in l.text) else "qwen3_voicedesign"

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
        existing_char = resolved_cast_map.get(s_key) or _resolve_character(speaker)

        if target_eng != last_used_model:
            model_swaps += 1
            last_used_model = target_eng

        if target_eng == "qwen3_clone" and existing_char is not None:
            take = _run_synthesis_and_save(
                SynthesizeRequest(
                    model_id="qwen3_clone",
                    character_id=existing_char["id"],
                    text=line.text.strip(),
                    speed=line_speed,
                    trim_silence=req.trim_silence,
                )
            )
            action = "reused_locked_character"
            instruct = existing_char.get("description", "")
        else:
            instruct = (
                line.instruct
                or cast_descriptions.get(s_key)
                or (existing_char["description"] if existing_char else None)
                or _find_preset_instruct(speaker, for_maya=(target_eng == "maya1"))
                or f"Expressive character voice for {speaker}, clear diction, natural conversational pacing."
            )
            take = _run_synthesis_and_save(
                SynthesizeRequest(
                    model_id=target_eng,
                    text=line.text.strip(),
                    instruct=instruct,
                    speed=line_speed,
                    trim_silence=req.trim_silence,
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
        pause_ms = line.pause_after_ms if line.pause_after_ms is not None else req.default_pause_ms

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
            "character_id": char_obj["id"] if char_obj else None,
            "model_id": take["model_id"],
            "action": action,
            "text": line.text.strip(),
            "instruct": instruct_used,
            "speed": line_speed,
            "take_id": take["id"],
            "take_url": take["url"],
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
    new_speed = req.speed if req.speed is not None else float(target_seg.get("speed", 1.15))
    char_obj = _resolve_character(target_seg.get("character_id") or speaker)

    model_id = req.model_id or target_seg.get("model_id") or ("qwen3_clone" if char_obj else "qwen3_voicedesign")
    instruct = (
        req.instruct
        or target_seg.get("instruct")
        or (char_obj["description"] if char_obj else "")
        or f"Expressive game character voice for {speaker}."
    )

    synth_req = SynthesizeRequest(
        model_id=model_id,
        text=new_text,
        instruct=instruct,
        character_id=char_obj["id"] if (model_id == "qwen3_clone" and char_obj) else None,
        speed=new_speed,
        temperature=req.temperature,
    )
    new_take = _run_synthesis_and_save(synth_req)

    target_seg["text"] = new_text
    target_seg["instruct"] = instruct
    target_seg["model_id"] = model_id
    target_seg["speed"] = new_speed
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


app.mount("/audio/scenes", StaticFiles(directory=str(SCENES_DIR)), name="scenes")
app.mount("/audio/takes", StaticFiles(directory=str(TAKES_DIR)), name="takes")
app.mount("/audio/characters", StaticFiles(directory=str(CHARACTERS_DIR)), name="characters")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    index_file = STATIC_DIR / "index.html"
    return index_file.read_text(encoding="utf-8")
