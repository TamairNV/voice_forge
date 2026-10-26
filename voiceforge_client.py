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
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} on {path}: {err_body}") from e

    def status(self) -> Dict[str, Any]:
        """Get full studio status, active model, VRAM telemetry, characters, and takes."""
        return self._request_json("GET", "/api/status")

    def get_llm_guide(self) -> Dict[str, Any]:
        """Fetch the full LLM System Prompt, 7-Dimension Qwen3 formula, Maya-1 `<description>` rules, and script syntax."""
        return self._request_json("GET", "/api/llm-guide")

    def optimize_prompt(
        self,
        concept: str,
        name: str = "Character",
        model_id: str = "qwen3_voicedesign",
        dialogue_text: str = "",
    ) -> Dict[str, Any]:
        """Expand a brief character idea into research-optimized Qwen3-TTS (7-Dimension) and Maya-1 (`<description>`) prompts."""
        return self._request_json(
            "POST",
            "/api/prompts/optimize",
            {
                "concept": concept,
                "name": name,
                "model_id": model_id,
                "dialogue_text": dialogue_text,
            },
        )

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
        speed: float = 1.0,
        trim_silence: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        output_wav: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Generate speech from a character/voice description using 'qwen3_voicedesign' or 'maya1'.
        Defaults to pure, unprocessed neural output (`speed=1.0, trim_silence=False`).
        Optionally saves the WAV file to `output_wav`.
        """
        payload: Dict[str, Any] = {
            "model_id": model_id,
            "text": text,
            "instruct": instruct,
            "speed": speed,
            "trim_silence": trim_silence,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        if seed is not None:
            payload["seed"] = seed
        res = self._request_json("POST", "/api/synthesize", payload)
        if output_wav:
            self.download_take(res["take"]["url"], output_wav)
        return res["take"]

    def design_and_lock_character(
        self,
        name: str,
        instruct: str,
        sample_text: str = "Keep your eyes sharp. Nothing in this sector is what it seems, and we only get one shot at this.",
        design_model_id: str = "qwen3_voicedesign",
        speed: float = 1.0,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        One-shot LLM helper:
        1. Invents a brand new voice from `instruct` (saved at pure 1.0x unprocessed quality)
        2. Locks it as a persistent NPC character under `name`
        3. Returns the locked character dict (with `id` for `speak_as_character`)
        """
        payload: Dict[str, Any] = {
            "name": name,
            "instruct": instruct,
            "sample_text": sample_text,
            "design_model_id": design_model_id,
            "speed": speed,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if seed is not None:
            payload["seed"] = seed
        return self._request_json(
            "POST",
            "/api/characters/design-and-lock",
            payload,
        )["character"]

    def speak_as_character(
        self,
        character_id_or_name: str,
        text: str,
        speed: float = 1.0,
        trim_silence: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        prev_take_id: Optional[str] = None,
        output_wav: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Speak a new dialogue line using a locked NPC character voice (`qwen3_clone`) with zero vocal drift.
        Pass `prev_take_id` to chain the previous line's audio + transcript into the context window!
        """
        payload: Dict[str, Any] = {
            "model_id": "qwen3_clone",
            "character_id": character_id_or_name,
            "text": text,
            "speed": speed,
            "trim_silence": trim_silence,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        if seed is not None:
            payload["seed"] = seed
        if prev_take_id is not None:
            payload["prev_take_id"] = prev_take_id
        res = self._request_json("POST", "/api/synthesize", payload)
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
        speed: float = 1.0,
        trim_silence: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        auto_lock_new_characters: bool = True,
        context_continuity: bool = True,
        output_wav: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Render a multi-character script into a stitched 24kHz `.wav` scene:
        - Automatically reuses existing locked characters by name (`qwen3_clone`).
        - Automatically designs & locks brand-new characters when first encountered (`auto_lock_new_characters=True`).
        - Chains each character's previous line audio + transcript (`context_continuity=True`) so lines flow naturally!
        - Synthesizes each line and stitches them together with `default_pause_ms` (or per-line `pause_after_ms`).
        """
        payload: Dict[str, Any] = {
            "title": title,
            "cast": cast or {},
            "lines": lines or [],
            "script_text": script_text,
            "default_pause_ms": default_pause_ms,
            "speed": speed,
            "trim_silence": trim_silence,
            "auto_lock_new_characters": auto_lock_new_characters,
            "context_continuity": context_continuity,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        if seed is not None:
            payload["seed"] = seed
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
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Surgically re-roll (or edit the text/speed/pause/engine/sampling params of) a single line `line_index` (1-based)
        in an existing scene and immediately re-stitch the master `.wav`, `.srt`, and `manifest.json`.
        """
        payload: Dict[str, Any] = {"line_index": line_index}
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
        if top_p is not None:
            payload["top_p"] = top_p
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        if seed is not None:
            payload["seed"] = seed
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
