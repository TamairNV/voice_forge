"""
Renders side_hustle_milk_1 v3: a lower-pitched, walking-stereotype Essex Karen
(MLM "boss babe", does her own research, everyone is jealous of her) on both engines:
  1. Qwen3-TTS: design & lock the voice once, then speak every line with qwen3_clone
  2. Maya-1: same script with inline acting tags
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from voiceforge_client import VoiceForgeClient

OUT_DIR = Path(__file__).parent.resolve()
NAME = "Kayleigh (Essex Karen)"

KAREN_INSTRUCT_QWEN = (
    "A smug, self-assured 28-year-old female MLM boss-babe and Karen stereotype who trusts anyone confident "
    "and thinks everyone is jealous of her, with a broad Essex estuary English accent, low alto chest-voice pitch with no shrillness, "
    "deep-for-a-woman husky smoky vape-raspy vocal texture, loud carrying phone-call voice, unhurried know-it-all cadence "
    "that snaps defensive, close-mic studio game dialogue."
)
GOLDEN_SAMPLE = (
    "Honestly babe, I've done my own research, alright? I'm my own boss now, I don't need anyone's permission, "
    "and if people can't handle that, that's literally a them problem."
)

KAREN_BASE_MAYA = (
    "Smug MLM boss-babe Karen, Female voice in her late 20s with an Essex English accent, "
    "low-mid pitch, husky slightly raspy timbre, conversational pacing"
)

# (emotion, text, maya_text, pause_after_ms) — pauses follow the original beat sheet
LINES = [
    ("casual smug", "Right, don't panic, babe. I've found a new side hustle.",
     "Right, don't panic, babe. I've found a new side hustle.", 1500),
    ("casual know-it-all", "Turns out you can sell your breast milk. They call it liquid gold. Look it up.",
     "Turns out you can sell your breast milk. <chuckle> They call it liquid gold. Look it up.", 2500),
    ("defensive", "No. This is nothing like the Herbalife thing. Or the crypto. That was different.",
     "<gasp> No. This is nothing like the Herbalife thing... or the crypto. That was different.", 2000),
    ("defensive, justifying", "Yes, I met him behind the big Tesco... but he's legit. He's got a van.",
     "<sigh> Yes, I met him behind the big Tesco... but he's legit. He's got a van.", 300),
    ("proud", "And he gave me a business card. With a QR code, Debbie. A QR code.",
     "And he gave me a business card. With a QR code, Debbie. <excited> A QR code.", 3000),
    ("indignant", "Wow. You know what? You've always been jealous. You just don't want me to succeed.",
     "<angry> Wow. You know what? You've always been jealous. You just don't want me to succeed.", 0),
]


def main():
    client = VoiceForgeClient("http://127.0.0.1:7860")

    print(f"1) Designing & locking '{NAME}'...")
    char = client.design_and_lock_character(
        name=NAME,
        instruct=KAREN_INSTRUCT_QWEN,
        sample_text=GOLDEN_SAMPLE,
        design_model_id="qwen3_voicedesign",
        speed=1.08,
    )
    print(f"   -> Locked {char['id']} ({char['name']})")

    print("\n2) Rendering Qwen3 (locked clone) version...")
    scene_qwen = client.render_scene(
        title="side_hustle_milk_1 v3 (Qwen3 locked Essex Karen)",
        default_pause_ms=300,
        speed=1.08,
        auto_lock_new_characters=False,
        lines=[
            {"speaker": NAME, "model_id": "qwen3_clone", "character_id": char["id"],
             "text": text, "pause_after_ms": pause}
            for _, text, _, pause in LINES
        ],
        output_wav=str(OUT_DIR / "side_hustle_milk_1_v3_qwen3.wav"),
    )
    print(f"   -> Saved {scene_qwen['file_path']} ({scene_qwen['duration_sec']}s)")

    print("\n3) Rendering Maya-1 acting version...")
    scene_maya = client.render_scene(
        title="side_hustle_milk_1 v3 (Maya-1 Essex Karen)",
        default_pause_ms=300,
        speed=1.08,
        auto_lock_new_characters=False,
        lines=[
            {"speaker": "Kayleigh_Maya", "model_id": "maya1",
             "instruct": f"{KAREN_BASE_MAYA}, {emotion} tone at medium-high intensity.",
             "text": maya_text, "pause_after_ms": pause}
            for emotion, _, maya_text, pause in LINES
        ],
        output_wav=str(OUT_DIR / "side_hustle_milk_1_v3_maya1.wav"),
    )
    print(f"   -> Saved {scene_maya['file_path']} ({scene_maya['duration_sec']}s)")


if __name__ == "__main__":
    main()
