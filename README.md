# VoiceForge Studio — Local Apple Silicon Voice Design, Acting & Cloning

**VoiceForge Studio** is a local 24kHz Text-to-Speech (TTS), Voice Design, Expressive Acting, and Zero-Shot Character Cloning studio built natively for **Apple Silicon (M-series Macs)** using **Apple's Metal MLX framework**.

It includes both a **100vh zero-scroll dark-mode Web Studio** (`http://127.0.0.1:7860`) and a **full REST + OpenAI-compatible API** (`http://127.0.0.1:7860/docs`) so another LLM, AI agent, or game engine can control every feature programmatically.

---

## 🚀 Supported Local Models (One-Click or API Hot-Swappable)

| Engine ID | Model Name | Hugging Face Repo | Precision / VRAM | Best Used For |
| :--- | :--- | :--- | :--- | :--- |
| `qwen3_voicedesign` | **Qwen3-TTS 1.7B — Voice Design** | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | `bf16` (~4.5 GB) | Inventing brand-new studio voices from a natural-language description of age, gender, timbre, accent, and pacing (~2.0x real-time on M4 Pro). |
| `maya1` | **Maya-1 3.3B — Expressive Acting** | `nhe-ai/maya1-mlx-8Bit` + `snac_24khz` | `8-bit MLX` (~3.6 GB) | Character acting with voice descriptions + **20+ inline emotion tags** (`<laugh>`, `<sigh>`, `<whisper>`, `<gasp>`, `<angry>`, `<cry>`, `<snort>`). |
| `qwen3_clone` | **Qwen3-TTS 1.7B — Base Voice Lock** | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | `bf16` (~4.5 GB) | Zero-shot voice cloning that locks onto a saved Character Voice from your roster so 1,000+ NPC game dialogue lines never drift. |

*Note: Only one model is held in Apple Metal VRAM at a time. Swapping models automatically unloads the previous weights (`gc.collect()` + `mx.clear_cache()`) and loads the new engine in ~1.2 seconds.*

---

## 🛠️ Setup Instructions (macOS Apple Silicon)

### 1. Prerequisites
- **macOS** on Apple Silicon (M1 / M2 / M3 / M4, 16GB+ Unified Memory recommended)
- **Python 3.11+** and **ffmpeg** (installed via Homebrew)

### 2. One-Command Installation
Clone the repo and run `./setup.sh`. It creates the `.venv`, installs `mlx-audio` + FastAPI, and pre-caches the MLX model weights:

```bash
cd ~/voice_forge
chmod +x setup.sh start_studio.sh
./setup.sh
```

### 3. Start the Studio Server
```bash
./start_studio.sh
```
- **Web Studio UI**: [http://127.0.0.1:7860](http://127.0.0.1:7860)
- **Interactive Swagger / OpenAPI Docs**: [http://127.0.0.1:7860/docs](http://127.0.0.1:7860/docs)
- **OpenAPI Spec (for LLM Tool Calling)**: [http://127.0.0.1:7860/openapi.json](http://127.0.0.1:7860/openapi.json)

---

## 🤖 Complete API Reference (For LLMs, AI Agents & Game Engines)

Every single feature in VoiceForge Studio is accessible over HTTP JSON or direct binary `.wav` streaming.

### Quick Summary of Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Quick health & active model check. |
| `GET` | `/api/status` | Full studio status: active model, Metal VRAM usage, models list, Maya-1 tags, presets, locked characters, and recent takes. |
| `GET` | `/api/models` | List the 3 engines (`qwen3_voicedesign`, `maya1`, `qwen3_clone`) and which one is loaded in VRAM. |
| `POST` | `/api/model/load` | Hot-swap the active model in Metal VRAM (`{"model_id": "maya1"}`). |
| `GET` | `/api/tags` | List all 20 inline acting tags supported by `maya1`. |
| `GET` | `/api/presets` | List built-in game character presets. |
| `POST` | `/api/synthesize` | Synthesize speech and return JSON metadata (`take.id`, `take.url`, `take.file_path`, `metrics`). |
| `POST` | `/api/synthesize/wav` | Synthesize speech and return the **raw 24kHz `.wav` binary audio** directly in the response body. |
| `POST` | `/v1/audio/speech` | **OpenAI-compatible TTS endpoint**—works as a drop-in local replacement for OpenAI TTS clients. |
| `GET` | `/api/characters` | List all locked NPC character voices in the Character Voice Locker. |
| `POST` | `/api/characters/design-and-lock` | **All-in-One LLM Endpoint**: Designs a new voice from a text description AND automatically saves it as a locked character in one call. |
| `POST` | `/api/characters/from-take` | Lock an existing `take_id` into the Character Voice Locker. |
| `POST` | `/api/characters/upload` | Upload an external `.wav` file + transcript (`multipart/form-data`) to create a locked character. |
| `DELETE` | `/api/characters/{char_id}` | Delete a locked character. |
| `POST` | `/api/scenes/render` | **Multi-Character Scene Director**: Takes a script (`script_text` or `lines` + `cast`), reuses existing characters by name or creates/locks new ones, generates every segment, and stitches them into a single master `.wav` scene with timeline timestamps. |
| `POST` | `/api/scenes/render/wav` | Same as `/api/scenes/render`, returning the stitched master 24kHz `.wav` binary file directly. |
| `POST` | `/api/scenes/stitch` | Stitch an ordered list of existing `take_ids` into a combined `.wav` scene. |
| `GET` | `/api/scenes` | List all stitched multi-character scenes. |
| `DELETE` | `/api/scenes/{scene_id}` | Delete a stitched scene. |
| `GET` | `/api/takes` | List recent generated audio takes. |
| `DELETE` | `/api/takes/{take_id}` | Delete a generated take. |

---

### Example 1: Design a Voice from a Description (`POST /api/synthesize`)

```bash
curl -X POST http://127.0.0.1:7860/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3_voicedesign",
    "instruct": "An elderly, weathered male dwarven blacksmith in his 60s with a deep, gravelly baritone voice and a Scottish accent.",
    "text": "Keep your coin, traveler. That blade was forged in the old fires.",
    "speed": 1.15,
    "trim_silence": true
  }'
```

**Want the `.wav` file directly in one command?** Call `/api/synthesize/wav`:
```bash
curl -X POST http://127.0.0.1:7860/api/synthesize/wav \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3_voicedesign",
    "instruct": "A sharp, witty 28-year-old female rogue with a sleek British accent, confident and sarcastic.",
    "text": "Oh, brilliant plan. Let us walk right through the main security grid.",
    "speed": 1.15
  }' --output rogue_line.wav
```

---

### Example 2: Expressive Acting with Maya-1 Inline Tags (`model_id: "maya1"`)

Supported inline tags in `text`:
`<laugh>`, `<laugh_harder>`, `<chuckle>`, `<giggle>`, `<sigh>`, `<exhale>`, `<gasp>`, `<gulp>`, `<snort>`, `<whisper>`, `<angry>`, `<sarcastic>`, `<mischievous>`, `<excited>`, `<curious>`, `<disappointed>`, `<appalled>`, `<cry>`, `<scream>`, `<sing>`

```bash
curl -X POST http://127.0.0.1:7860/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "maya1",
    "instruct": "Sarcastic female rogue character, Female voice in her late 20s with a British accent, brisk pacing, sarcastic tone.",
    "text": "Oh, brilliant plan! <laugh> I am sure the automated turrets will not mind at all <sigh>.",
    "speed": 1.15,
    "trim_silence": true
  }'
```

---

### Example 3: All-in-One LLM Game Workflow ("Design Once, Clone Forever")

If another LLM is generating characters and dialogue for your game, it can:
1. Call **`POST /api/characters/design-and-lock`** once when creating a new NPC.
2. Call **`POST /api/synthesize`** with `model_id: "qwen3_clone"` and `character_id: "<name or id>"` for all future lines of that NPC so the voice never drifts!

```bash
# Step 1: Invent a voice from description AND lock it as a persistent character
curl -X POST http://127.0.0.1:7860/api/characters/design-and-lock \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Commander Vane",
    "instruct": "A battle-hardened 45-year-old male starship commander, deep authoritative American baritone, brisk tactical pacing.",
    "sample_text": "All stations, lock shields to maximum. Nobody fires until I give the signal.",
    "design_model_id": "qwen3_voicedesign",
    "speed": 1.15
  }'

# Step 2: Generate any number of new dialogue lines using Commander Vane's locked voice (by name or ID!)
curl -X POST http://127.0.0.1:7860/api/synthesize/wav \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3_clone",
    "character_id": "Commander Vane",
    "text": "They breached the lower hangar! Reroute auxiliary power to the starboard thrusters now!",
    "speed": 1.20
  }' --output vane_line_2.wav
```

---

### Example 4: Using the Built-In Python SDK (`voiceforge_client.py`)

Any Python script or LLM agent can import [`voiceforge_client.py`](voiceforge_client.py) (zero external dependencies):

```python
from voiceforge_client import VoiceForgeClient

client = VoiceForgeClient("http://127.0.0.1:7860")

# 1. Design & lock a new game character in one call
char = client.design_and_lock_character(
    name="Lyra the Alchemist",
    instruct=(
        "A cheerful, energetic young female alchemist in her 20s, bright"
        " American accent, fast excited pacing."
    ),
)

# 2. Speak unlimited consistent lines as that character and save directly to .wav
take = client.speak_as_character(
    character_id_or_name=char["id"],
    text=(
        "Careful with that blue potion! It's still bubbling!"
    ),
    speed=1.15,
    output_wav="dialogue/lyra_01.wav",
)
print("Saved WAV:", take["file_path"])
```

---

### Example 5: OpenAI-Compatible Endpoint (`POST /v1/audio/speech`)

Works with any tool or LLM framework expecting OpenAI's `/v1/audio/speech` format:

```bash
curl -X POST http://127.0.0.1:7860/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_voicedesign",
    "voice": "Torvin the Ironwright (Dwarven Blacksmith)",
    "input": "You want a greatsword? Bring me three ingots of star-iron first.",
    "speed": 1.15
  }' --output speech.wav
```
*(If `voice` matches a saved character name/ID, it automatically clones that character; otherwise it uses `instructions` or `voice` as a natural-language voice description!)*

---

### Example 6: Multi-Character Scene & Script Stitcher (`POST /api/scenes/render`)

You (or another LLM) can pass an entire multi-character game script in a single API call.
- **Reuses Existing Characters**: If a speaker name (e.g., `"Torvin"`) already exists in your Character Voice Locker, it automatically uses `qwen3_clone` with their locked voice so they sound 100% identical across every scene.
- **Auto-Creates & Locks New Characters**: If a speaker name (e.g., `"Nyx"`) is new, it generates their voice from their description on their first line, locks them into your Character Voice Locker (`auto_lock_new_characters: true`), and reuses that exact voice for all of their subsequent lines in the scene and future scenes!
- **Stitches Master Scene + Individual Takes**: Concatenates all segments with configurable turn-taking silence (`default_pause_ms` or per-line `pause_after_ms`) into a single master 24kHz `.wav` scene and returns start/end timestamps for subtitles.

#### Option A: Plain-Text Script (`script_text`)
```bash
curl -X POST http://127.0.0.1:7860/api/scenes/render \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Blacksmith Confrontation",
    "default_pause_ms": 280,
    "speed": 1.15,
    "script_text": "Torvin: Keep your coin, traveler. That blade was forged in the old fires.\nNyx (A sharp, witty 28-year-old female rogue with a sleek British accent, confident and sarcastic tone, brisk pacing): Oh, come on Torvin! Everyone in this city has a price.\n[Pause: 400ms]\nTorvin: Not me, girl. Get out of my forge.\nNyx: Fine, keep the sword. I only came for the map."
  }'
```

#### Option B: Structured JSON (`cast` + `lines`)
```bash
curl -X POST http://127.0.0.1:7860/api/scenes/render \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Bridge Ambush",
    "default_pause_ms": 250,
    "speed": 1.15,
    "cast": {
      "Nyx": "A sharp, witty 28-year-old female rogue with a sleek British accent, confident and sarcastic tone, brisk pacing.",
      "Captain Vane": "A battle-hardened 45-year-old male guard captain, deep authoritative American baritone, brisk tactical pacing."
    },
    "lines": [
      {"speaker": "Captain Vane", "text": "Halt! Nobody crosses the upper bridge tonight.", "pause_after_ms": 300},
      {"speaker": "Nyx", "text": "Good thing I was planning on taking the rooftops then."},
      {"speaker": "Captain Vane", "text": "Archers, lock onto her now!"}
    ]
  }'
```
*(Want the stitched `.wav` file returned directly as binary audio? Send the exact same payload to `POST /api/scenes/render/wav` with `--output scene.wav`!)*
