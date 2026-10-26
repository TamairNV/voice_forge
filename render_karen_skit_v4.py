"""
Renders side_hustle_milk_1 v4 from the "loud, nasal, entitled mid-20s phone talker" brief,
with the emotional arc (breezy -> defensive -> smug -> indignant) written into each line's prompt:
  1. Qwen3 VoiceDesign per line (follows the arc; the voice may drift slightly between lines)
  2. Qwen3 locked clone (one consistent voice; the arc comes from punctuation only)
  3. Maya-1 with inline acting tags
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from voiceforge_client import VoiceForgeClient

OUT_DIR = Path(__file__).parent.resolve()
NAME = "Kayleigh v4 (Loud Nasal Karen)"

QWEN_BASE = (
    "A loud, nasal, entitled woman in her mid-20s with an Essex English accent, speaking on her phone far too "
    "loudly in public, low alto chest-voice baseline pitch with no shrillness, deep-for-a-woman husky tone with heavy vocal fry, "
    "bratty nasal timbre, close-mic studio game dialogue"
)
GOLDEN_SAMPLE = (
    "Yeah, no, I hear you, babe, but honestly I have got it all sorted, alright? I just think some people "
    "get a bit weird when they see you actually doing well for yourself."
)

MAYA_BASE = (
    "Loud entitled phone-talker, Female voice in her mid-20s with an Essex English accent, "
    "low-mid pitch with heavy vocal fry, nasal bratty timbre"
)

# (arc direction, qwen text, maya text, pause_after_ms) — pauses follow the original beat sheet
LINES = [
    ("casual and breezy, relaxed pacing, lazy vocal fry at the ends of phrases",
     "Right, don't panic, babe. I've found a new side hustle.",
     "Right, don't panic, babe. I've found a new side hustle.", 1500),
    ("casual and breezy, offhand and matter-of-fact",
     "Turns out you can sell your breast milk. They call it liquid gold. Look it up.",
     "Turns out you can sell your breast milk. They call it liquid gold. Look it up.", 2500),
    ("defensive and fast, clipped rapid delivery, sharp annoyed pitch rise",
     "No. No! This is nothing like the Herbalife thing. Or the crypto. That was different!",
     "<gasp> No. No! This is nothing like the Herbalife thing... or the crypto. That was different!", 2000),
    ("defensive and fast, flustered and justifying, dramatic sigh, words tumbling out",
     "Ugh, yes, I met him behind the big Tesco... but he's legit. He's got a van.",
     "<sigh> Yes, I met him behind the big Tesco... but he's legit. He's got a van.", 300),
    ("proud and smug, slow self-satisfied drawl with vocal fry",
     "And he gave me a business card. With a QR code, Debbie. A QR code.",
     "And he gave me a business card. <chuckle> With a QR code, Debbie. A QR code.", 3000),
    ("indignant and offended, big dramatic pitch rise, wounded and loud",
     "Wow. You know what? You've always been jealous. You just don't want me to succeed!",
     "<sigh> Wow. You know what? You've always been jealous. <angry> You just don't want me to succeed!", 0),
]


def main():
    client = VoiceForgeClient("http://127.0.0.1:7860")

    print("1) Rendering Qwen3 VoiceDesign per-line version...")
    scene = client.render_scene(
        title="side_hustle_milk_1 v4 (Qwen3 design, per-line arc)",
        default_pause_ms=300,
        speed=1.1,
        auto_lock_new_characters=False,
        lines=[
            {"speaker": "Kayleigh_Design", "model_id": "qwen3_voicedesign",
             "instruct": f"{QWEN_BASE}, {arc}.", "text": text, "pause_after_ms": pause}
            for arc, text, _, pause in LINES
        ],
        output_wav=str(OUT_DIR / "side_hustle_milk_1_v4_qwen3_design.wav"),
    )
    print(f"   -> {scene['duration_sec']}s")

    print("\n2) Designing & locking voice, then rendering Qwen3 clone version...")
    char = client.design_and_lock_character(
        name=NAME,
        instruct=f"{QWEN_BASE}, relaxed breezy delivery.",
        sample_text=GOLDEN_SAMPLE,
        design_model_id="qwen3_voicedesign",
        speed=1.1,
    )
    scene = client.render_scene(
        title="side_hustle_milk_1 v4 (Qwen3 locked clone)",
        default_pause_ms=300,
        speed=1.1,
        auto_lock_new_characters=False,
        lines=[
            {"speaker": NAME, "model_id": "qwen3_clone", "character_id": char["id"],
             "text": text, "pause_after_ms": pause}
            for _, text, _, pause in LINES
        ],
        output_wav=str(OUT_DIR / "side_hustle_milk_1_v4_qwen3_clone.wav"),
    )
    print(f"   -> {char['id']}, {scene['duration_sec']}s")

    print("\n3) Rendering Maya-1 version...")
    scene = client.render_scene(
        title="side_hustle_milk_1 v4 (Maya-1)",
        default_pause_ms=300,
        speed=1.1,
        auto_lock_new_characters=False,
        lines=[
            {"speaker": "Kayleigh_Maya", "model_id": "maya1",
             "instruct": f"{MAYA_BASE}, {arc} tone at high intensity.", "text": maya_text, "pause_after_ms": pause}
            for arc, _, maya_text, pause in LINES
        ],
        output_wav=str(OUT_DIR / "side_hustle_milk_1_v4_maya1.wav"),
    )
    print(f"   -> {scene['duration_sec']}s")


if __name__ == "__main__":
    main()
