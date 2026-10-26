"""
Renders side_hustle_milk_1 v5 ("Gary" script) with the v4 loud-nasal-Karen voice on:
  1. Qwen3 VoiceDesign per line (same base description every line, short arc cue appended)
  2. Maya-1 with inline acting tags
The locked-clone version is skipped: it had the worst consistency in v4.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from voiceforge_client import VoiceForgeClient
from render_karen_skit_v4 import MAYA_BASE, QWEN_BASE

OUT_DIR = Path(__file__).parent.resolve()

# Script gaps: "..." = 1.2s, "...." = 1.6s, "......" = 2.5s, same paragraph = 0.25s
# (arc cue, qwen text, maya text, pause_after_ms)
LINES = [
    ("breezy and reassuring",
     "Mum, don't worry. I've sorted it. I've got a new side hustle.",
     "Mum, don't worry. I've sorted it. I've got a new side hustle.", 1200),
    ("casual and matter-of-fact",
     "I'm selling my breast milk.",
     "I'm selling my breast milk.", 2500),
    ("defensive and fast, sharp annoyed pitch rise",
     "No. No, this is NOT like last time.",
     "<sigh> No. No, this is NOT like last time.", 1200),
    ("defensive and condescending, explaining slowly",
     "That was a pyramid scheme. This is a business.",
     "That was a pyramid scheme. This is a business.", 1600),
    ("flustered and justifying",
     "Okay, yes, I met him behind a Tesco. But he's legit.",
     "Okay, yes, I met him behind a Tesco. But he's legit.", 250),
    ("proud and triumphant",
     "He's got a business card!",
     "<excited> He's got a business card!", 1200),
    ("proud, slightly less sure",
     "It just says Gary. And a phone number.",
     "It just says Gary... and a phone number.", 1200),
    ("smug self-satisfied drawl with vocal fry",
     "Very minimalist. Very professional.",
     "Very minimalist. <chuckle> Very professional.", 2500),
    ("breezy and scheming",
     "Anyway, he's paying forty quid a pint, so... quick question.",
     "Anyway, he's paying forty quid a pint, so... quick question.", 250),
    ("innocently curious, totally serious",
     "How long does it take to have a baby?",
     "<curious> How long does it take to have a baby?", 0),
]


def main():
    client = VoiceForgeClient("http://127.0.0.1:7860")

    print("1) Rendering Qwen3 VoiceDesign version...")
    scene = client.render_scene(
        title="side_hustle_milk_1 v5 (Qwen3 design)",
        default_pause_ms=300,
        speed=1.1,
        auto_lock_new_characters=False,
        lines=[
            {"speaker": "Kayleigh_Design", "model_id": "qwen3_voicedesign",
             "instruct": f"{QWEN_BASE}, {arc}.", "text": text, "pause_after_ms": pause}
            for arc, text, _, pause in LINES
        ],
        output_wav=str(OUT_DIR / "side_hustle_milk_1_v5_qwen3.wav"),
    )
    print(f"   -> {scene['duration_sec']}s")

    print("\n2) Rendering Maya-1 version...")
    scene = client.render_scene(
        title="side_hustle_milk_1 v5 (Maya-1)",
        default_pause_ms=300,
        speed=1.1,
        auto_lock_new_characters=False,
        lines=[
            {"speaker": "Kayleigh_Maya", "model_id": "maya1",
             "instruct": f"{MAYA_BASE}, {arc} tone at high intensity.", "text": maya_text, "pause_after_ms": pause}
            for arc, _, maya_text, pause in LINES
        ],
        output_wav=str(OUT_DIR / "side_hustle_milk_1_v5_maya1.wav"),
    )
    print(f"   -> {scene['duration_sec']}s")


if __name__ == "__main__":
    main()
