"""
End-to-End Test for the VoiceForge Game Skit & Cutscene Pipeline.
Run with:
    ./.venv/bin/python test_skit_pipeline.py
"""

from voiceforge_client import VoiceForgeClient


def run_test():
    client = VoiceForgeClient("http://127.0.0.1:7860")

    print("1) Rendering multi-character skit with [Overlap: 220ms] interruption...")
    script = """
    Nyx (sharp witty 28-year-old female rogue, crisp British accent, brisk pacing) [1.20x]: The security relay is cycling down, just give me ten more seconds on this terminal—
    [Overlap: 220ms]
    Torvin (elderly weathered Scottish dwarf blacksmith, deep gravelly baritone) [1.25x]: Ten seconds?! The iron golems are already coming down the corridor!
    Nyx [1.20x]: Relax, old man! Pull the red lever now!
    """
    scene = client.render_scene(
        title="Vault Laser Grid Interruption",
        script_text=script,
        default_pause_ms=220,
        speed=1.15,
        output_wav="test_scene_master.wav",
    )
    print(f"   -> Scene ID:     {scene['id']}")
    print(f"   -> Duration:     {scene['duration_sec']}s (Model swaps: {scene.get('model_swaps')})")
    for seg in scene["segments"]:
        print(f"      Line #{seg['index']} [{seg['start_sec']}s - {seg['end_sec']}s] {seg['speaker']}: {seg['text']}")

    print("\n2) Surgical Single-Line Re-Roll (updating Line #3 without re-rendering Lines #1 & #2)...")
    updated_scene = client.reroll_scene_line(
        scene_id=scene["id"],
        line_index=3,
        text="Relax, old man! Pull the red lever right now!",
        speed=1.25,
    )
    print(f"   -> Re-stitched in {updated_scene['elapsed_sec']}s | New Duration: {updated_scene['duration_sec']}s")

    print("\n3) Fetching generated SubRip (.SRT) subtitles...")
    srt_text = client.get_scene_subtitles_srt(scene["id"])
    print(srt_text)

    print("4) Downloading ready-to-import Game Engine Cutscene Bundle (.zip)...")
    zip_path = client.download_scene_bundle(scene["id"], "test_scene_bundle.zip")
    print(f"   -> Saved bundle: {zip_path.resolve()}")


if __name__ == "__main__":
    run_test()
