"""
VoiceForge Python SDK / Client for LLMs, AI Agents, and Game Scripts.
Zero external dependencies (uses standard library `urllib.request`) so any script or LLM agent can import it directly.
"""

import json
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


class VoiceForgeClient:
    """Simple Python client for controlling VoiceForge Studio via its local REST API."""

    def __init__(self, base_url: str = "http://127.0.0.1:7860"):
        self.base_url = base_url.rstrip("/")

    def _request_json(self, method: str, path: str, payload: Optional[dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def status(self) -> Dict[str, Any]:
        """Get full studio status, active model, VRAM telemetry, characters, and takes."""
        return self._request_json("GET", "/api/status")

    def list_models(self) -> List[Dict[str, Any]]:
        """List available TTS engines ('qwen3_voicedesign', 'maya1', 'qwen3_clone')."""
        return self._request_json("GET", "/api/models")["models"]

    def swap_model(self, model_id: str) -> Dict[str, Any]:
        """Hot-swap the active TTS model in Apple Metal VRAM."""
        return self._request_json("POST", "/api/model/load", {"model_id": model_id})

    def list_characters(self) -> List[Dict[str, Any]]:
        """List all locked NPC character voices in the Character Voice Locker."""
        return self._request_json("GET", "/api/characters")["characters"]

    def design_voice(
        self,
        text: str,
        instruct: str,
        model_id: str = "qwen3_voicedesign",
        speed: float = 1.15,
        trim_silence: bool = True,
        output_wav: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Generate speech from a character/voice description using 'qwen3_voicedesign' or 'maya1'.
        Optionally saves the WAV file to `output_wav`.
        """
        res = self._request_json(
            "POST",
            "/api/synthesize",
            {
                "model_id": model_id,
                "text": text,
                "instruct": instruct,
                "speed": speed,
                "trim_silence": trim_silence,
            },
        )
        if output_wav:
            self.download_take(res["take"]["url"], output_wav)
        return res["take"]

    def design_and_lock_character(
        self,
        name: str,
        instruct: str,
        sample_text: str = "Keep your eyes sharp. Nothing in this sector is what it seems, and we only get one shot at this.",
        design_model_id: str = "qwen3_voicedesign",
        speed: float = 1.15,
    ) -> Dict[str, Any]:
        """
        One-shot LLM helper:
        1. Invents a brand new voice from `instruct`
        2. Locks it as a persistent NPC character under `name`
        3. Returns the locked character dict (with `id` for `speak_as_character`)
        """
        return self._request_json(
            "POST",
            "/api/characters/design-and-lock",
            {
                "name": name,
                "instruct": instruct,
                "sample_text": sample_text,
                "design_model_id": design_model_id,
                "speed": speed,
            },
        )["character"]

    def speak_as_character(
        self,
        character_id_or_name: str,
        text: str,
        speed: float = 1.15,
        trim_silence: bool = True,
        output_wav: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Speak a new dialogue line using a locked NPC character voice (`qwen3_clone`) with zero vocal drift.
        """
        res = self._request_json(
            "POST",
            "/api/synthesize",
            {
                "model_id": "qwen3_clone",
                "character_id": character_id_or_name,
                "text": text,
                "speed": speed,
                "trim_silence": trim_silence,
            },
        )
        if output_wav:
            self.download_take(res["take"]["url"], output_wav)
        return res["take"]

    def render_scene(
        self,
        title: str = "Game Cutscene",
        script_text: Optional[str] = None,
        lines: Optional[List[Dict[str, Any]]] = None,
        cast: Optional[Dict[str, str]] = None,
        default_pause_ms: int = 280,
        speed: float = 1.15,
        trim_silence: bool = True,
        auto_lock_new_characters: bool = True,
        output_wav: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Render a multi-character script into a stitched 24kHz `.wav` scene:
        - Automatically reuses existing locked characters by name (`qwen3_clone`).
        - Automatically designs & locks brand-new characters when first encountered (`auto_lock_new_characters=True`).
        - Synthesizes each line and stitches them together with `default_pause_ms` (or per-line `pause_after_ms`).
        """
        payload = {
            "title": title,
            "cast": cast or {},
            "lines": lines or [],
            "script_text": script_text,
            "default_pause_ms": default_pause_ms,
            "speed": speed,
            "trim_silence": trim_silence,
            "auto_lock_new_characters": auto_lock_new_characters,
        }
        res = self._request_json("POST", "/api/scenes/render", payload)
        scene = res["scene"]
        if output_wav:
            self.download_take(scene["url"], output_wav)
        return scene

    def render_scenes_batch(self, scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Render multiple multi-character skits/cutscenes in a single low-friction batch call.
        Characters created & locked in earlier skits are automatically reused in subsequent skits!
        """
        return self._request_json("POST", "/api/scenes/batch", {"scenes": scenes})

    def reroll_scene_line(
        self,
        scene_id: str,
        line_index: int,
        text: Optional[str] = None,
        instruct: Optional[str] = None,
        model_id: Optional[str] = None,
        speed: Optional[float] = None,
        pause_after_ms: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Surgically re-roll (or edit the text/speed/pause/engine of) a single line `line_index` (1-based)
        in an existing scene and immediately re-stitch the master `.wav`, `.srt`, and `manifest.json`.
        """
        payload = {"line_index": line_index}
        if text is not None:
            payload["text"] = text
        if instruct is not None:
            payload["instruct"] = instruct
        if model_id is not None:
            payload["model_id"] = model_id
        if speed is not None:
            payload["speed"] = speed
        if pause_after_ms is not None:
            payload["pause_after_ms"] = pause_after_ms
        if temperature is not None:
            payload["temperature"] = temperature
        return self._request_json("POST", f"/api/scenes/{scene_id}/reroll-line", payload)["scene"]

    def download_scene_bundle(self, scene_id: str, dest_zip_path: Union[str, Path]) -> Path:
        """
        Download a complete Game Engine Cutscene Bundle (`.zip`) containing:
          - `master_scene.wav`
          - `subtitles.srt`
          - `manifest.json`
          - `lines/001_<speaker>.wav`, `002_<speaker>.wav`, ...
        """
        return self.download_take(f"/api/scenes/{scene_id}/export.zip", dest_zip_path)

    def get_scene_subtitles_srt(self, scene_id: str) -> str:
        """Fetch the SubRip (`.srt`) subtitle file content for a rendered scene."""
        url = f"{self.base_url}/api/scenes/{scene_id}/subtitles.srt"
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8")

    def get_scene_manifest(self, scene_id: str) -> Dict[str, Any]:
        """Fetch the Unity / Unreal / Godot dialogue manifest JSON for a rendered scene."""
        return self._request_json("GET", f"/api/scenes/{scene_id}/manifest.json")

    def stitch_takes(
        self,
        take_ids: List[str],
        title: str = "Stitched Scene",
        pause_ms: int = 280,
        output_wav: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Stitch a list of existing `take_ids` into a single combined `.wav` scene."""
        res = self._request_json(
            "POST",
            "/api/scenes/stitch",
            {"title": title, "take_ids": take_ids, "pause_ms": pause_ms},
        )
        scene = res["scene"]
        if output_wav:
            self.download_take(scene["url"], output_wav)
        return scene

    def download_take(self, take_url: str, dest_path: Union[str, Path]) -> Path:
        """Download a generated take's WAV file to a custom path."""
        url = f"{self.base_url}{take_url}" if take_url.startswith("/") else take_url
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dest))
        return dest


if __name__ == "__main__":
    client = VoiceForgeClient()
    st = client.status()
    print(f"Connected to VoiceForge Studio! Active Model: {st['active_model_id']}")
    print(f"Locked Characters ({len(st['characters'])}): {[c['name'] for c in st['characters']]}")
