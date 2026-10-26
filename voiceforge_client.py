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
