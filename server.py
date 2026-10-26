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

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import List, Optional

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
STATIC_DIR = BASE_DIR / "static"

for d in (DATA_DIR, TAKES_DIR, CHARACTERS_DIR, STATIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

TAKES_JSON = DATA_DIR / "takes.json"
CHARACTERS_JSON = DATA_DIR / "characters.json"


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
    """Full studio state: active model, Metal VRAM usage, available models, Maya-1 tags, presets, characters, and recent takes."""
    status = engine.get_status()
    status["presets"] = CHARACTER_PRESETS
    status["characters"] = _read_json(CHARACTERS_JSON, [])
    status["takes"] = _read_json(TAKES_JSON, [])[:40]
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
    chars = _read_json(CHARACTERS_JSON, [])
    for c in chars:
        if c["id"] == character_id_or_name or c["name"].lower() == character_id_or_name.lower():
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


app.mount("/audio/takes", StaticFiles(directory=str(TAKES_DIR)), name="takes")
app.mount("/audio/characters", StaticFiles(directory=str(CHARACTERS_DIR)), name="characters")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    index_file = STATIC_DIR / "index.html"
    return index_file.read_text(encoding="utf-8")
