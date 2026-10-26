"""
Renders the `side_hustle_milk_1` entitled 20s Karen phone-call skit:
  1. Original Script (with exact phone-call pauses)
  2. Punched-Up / Improved Comedy Script (Locked Qwen3 Voice + Expressive Maya-1 Version)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from voiceforge_client import VoiceForgeClient


def main():
    client = VoiceForgeClient("http://127.0.0.1:7860")

    # 1. Design & lock the "Entitled 20s Karen / Delusional Girlboss" character voice
    # Pure 1.0x unprocessed Golden Reference Anchor + RVQ Timbre Lock
    karen_instruct_qwen = (
        "An ultra-entitled, bratty, self-important 25-year-old female Karen and delusional influencer stereotype "
        "with a sharp posh-tryhard British accent, medium-high slightly nasal pitch, crisp expressive vocal texture, "
        "snappy condescending cadence, dramatic defensive and judgmental attitude, close-mic studio game dialogue."
    )
    karen_golden_sample = (
        "Excuse me, do you actually know who you are talking to right now? I am building an empire from my phone, "
        "and frankly your negative energy is really ruining my manifestation today!"
    )

    # Remove any old Chloe characters so we test a fresh, pure 1.0x Golden Reference lock
    for c in client.list_characters():
        if "chloe" in c["name"].lower():
            client._request_json("DELETE", f"/api/characters/{c['id']}")

    print("1) Designing & Locking 'Chloe (Entitled 20s Karen)' into Character Voice Locker (Pure 1.0x Unprocessed)...")
    char = client.design_and_lock_character(
        name="Chloe (Entitled 20s Karen)",
        instruct=karen_instruct_qwen,
        sample_text=karen_golden_sample,
        design_model_id="qwen3_voicedesign",
        speed=1.0,
        seed=1042,
    )
    print(f"   -> Locked Character ID: {char['id']} ({char['name']})")

    # 2. Render Version 1: The User's Original Script (Pure 1.0x, RVQ Timbre Locked)
    original_skit = """
STORY: Chloe (Entitled 20s Karen)
NPC TYPE: Normal (female, 20s, entitled Karen)
TRIGGER: walking, phone call
─────────────────────────────────
[casual]     "Don't worry, I found a new side hustle."
[pause 1.5s]
[casual]     "I found out I can sell my breast milk."
[pause 2.5s]
[defensive]  "No, this isn't like last time!"
[pause 2.0s]
[defensive]  "Behind a Tesco, but he's legit."
[proud]      "He had a business card."
[pause 3.0s]
[indignant]  "You just don't want me to succeed!"
─────────────────────────────────
"""
    print("\n2) Rendering Original Skit with Locked Chloe Voice (Pure 1.0x, RVQ Timbre Lock)...")
    scene_orig = client.render_scene(
        title="side_hustle_milk_1 (Original Script - Locked Karen)",
        script_text=original_skit,
        default_pause_ms=300,
        speed=1.0,
        trim_silence=False,
        output_wav="/Users/tamair/voice_forge/side_hustle_milk_1_original.wav",
    )
    print(f"   -> Saved: {scene_orig['file_path']} ({scene_orig['duration_sec']}s)")
    for seg in scene_orig["segments"]:
        print(f"      Line #{seg['index']} [{seg['model_id']}, chained={seg.get('context_chained')}]: pitch={seg.get('pitch_hz_median')}Hz ({seg.get('voice_register')}) | \"{seg['text']}\"")

    # 3. Render Version 2: Punched-Up / Improved Game Skit Writing (Pure 1.0x, RVQ Timbre Locked Qwen3 Voice)
    improved_skit_qwen = """
STORY: Chloe (Entitled 20s Karen)
NPC TYPE: Entitled 20s Girlboss Karen (female, 20s)
TRIGGER: walking, loud speakerphone call in public
─────────────────────────────────
[casual]     "Babe, stop lecturing me about rent, okay? I literally just unlocked a passive income stream."
[pause 1.5s]
[casual]     "Liquid gold, hun. I'm selling my breast milk to bodybuilders."
[pause 2.5s]
[defensive]  "Excuse me?! No, this is nothing like the dog-breeding incident!"
[pause 2.0s]
[defensive]  "Okay, yes, we met behind the big Tesco by the recycling bins, but he is a serious investor!"
[pause 0.35s]
[proud]      "He literally handed me a business card. It was laminated, Sharon. Laminated!"
[pause 3.0s]
[indignant]  "Wow. Honestly, the toxic negativity from you right now? You just hate seeing a female founder win!"
─────────────────────────────────
"""
    print("\n3) Rendering Punched-Up / Improved Skit (Pure 1.0x, RVQ Timbre Locked Qwen3 Karen Voice)...")
    scene_improved = client.render_scene(
        title="side_hustle_milk_1 (Improved Comedy Script - Locked Qwen3)",
        script_text=improved_skit_qwen,
        default_pause_ms=300,
        speed=1.0,
        trim_silence=False,
        output_wav="/Users/tamair/voice_forge/side_hustle_milk_1_improved_qwen3.wav",
    )
    print(f"   -> Saved: {scene_improved['file_path']} ({scene_improved['duration_sec']}s)")
    for seg in scene_improved["segments"]:
        print(f"      Line #{seg['index']} [{seg['model_id']}, chained={seg.get('context_chained')}]: pitch={seg.get('pitch_hz_median')}Hz ({seg.get('voice_register')}) | \"{seg['text']}\"")

    # 4. Render Version 3: Hybrid Scene (Locked Qwen3 Clone for dialogue + SNAC Voice-Locked Maya-1 for <gasp>/<snort> lines)
    karen_base_maya = (
        "Entitled bratty 25-year-old female Karen influencer, Female voice in her 20s with a British accent, "
        "medium-high nasal pitch, sharp crisp vocal timbre, brisk condescending pacing, smug tone at high intensity."
    )
    print("\n4) Rendering Hybrid Skit (Locked Chloe Voice + SNAC Voice-Locked Maya-1 for <gasp>/<snort> lines)...")
    scene_maya = client.render_scene(
        title="side_hustle_milk_1 (Improved Comedy Script - Hybrid Locked + Maya-1 Tags)",
        default_pause_ms=300,
        speed=1.0,
        trim_silence=False,
        lines=[
            {
                "speaker": "Chloe (Entitled 20s Karen)",
                "text": "Babe, stop lecturing me about rent, okay? I literally just unlocked a passive income stream.",
                "pause_after_ms": 1500,
            },
            {
                "speaker": "Chloe (Entitled 20s Karen)",
                "text": "Liquid gold, hun. I'm selling my breast milk to bodybuilders.",
                "pause_after_ms": 2500,
            },
            {
                "speaker": "Chloe (Entitled 20s Karen)",
                "model_id": "maya1",
                "instruct": karen_base_maya,
                "text": "<gasp> Excuse me?! No, this is nothing like the dog-breeding incident!",
                "pause_after_ms": 2000,
            },
            {
                "speaker": "Chloe (Entitled 20s Karen)",
                "model_id": "maya1",
                "instruct": karen_base_maya,
                "text": "Okay, yes, we met behind the big Tesco by the recycling bins, <snort> but he is a serious investor!",
                "pause_after_ms": 350,
            },
            {
                "speaker": "Chloe (Entitled 20s Karen)",
                "text": "He literally handed me a business card. It was laminated, Sharon. Laminated!",
                "pause_after_ms": 3000,
            },
            {
                "speaker": "Chloe (Entitled 20s Karen)",
                "text": "Wow. Honestly, the toxic negativity from you right now? You just hate seeing a female founder win!",
                "pause_after_ms": 0,
            },
        ],
        output_wav="/Users/tamair/voice_forge/side_hustle_milk_1_improved_maya1.wav",
    )
    print(f"   -> Saved: {scene_maya['file_path']} ({scene_maya['duration_sec']}s)")
    for seg in scene_maya["segments"]:
        print(f"      Line #{seg['index']} [{seg['model_id']}, chained={seg.get('context_chained')}]: pitch={seg.get('pitch_hz_median')}Hz ({seg.get('voice_register')}) | \"{seg['text']}\"")


if __name__ == "__main__":
    main()
