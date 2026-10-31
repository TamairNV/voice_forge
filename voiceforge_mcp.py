#!/usr/bin/env python3
"""
VoiceForge Studio as tools for LLM agents: an MCP server (stdio) and a CLI in one stdlib-only file.

Every studio API operation becomes one tool, generated live from the running studio's /openapi.json,
so anything the web pages (/, /cast, /voices, /clone) can do, an agent can do too, and new endpoints
show up automatically.

  MCP (Claude Code / Claude Desktop / any MCP client):
      claude mcp add voiceforge -- python3 /Users/tamair/voice_forge/voiceforge_mcp.py

  CLI:
      python3 voiceforge_mcp.py list                      # every tool, one line each
      python3 voiceforge_mcp.py describe cast_produce     # its input schema
      python3 voiceforge_mcp.py call cast_produce '{"script": "Tamair: Hi!\\nBob: Hey.", "use_claude": false}'

Arguments are one flat JSON object: path parameters, query parameters and JSON body fields side by side.
Multipart uploads take a local path in `<field>_path` (e.g. `file_path`). Audio / zip / srt responses are
saved to $VOICEFORGE_OUT (default ./voiceforge_downloads) and the tool returns the saved path.

The studio must be running (./start_studio.sh). Base URL: $VOICEFORGE_URL (default http://127.0.0.1:7860).
"""

import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_URL = os.environ.get("VOICEFORGE_URL", "http://127.0.0.1:7860").rstrip("/")
OUT_DIR = Path(os.environ.get("VOICEFORGE_OUT", "voiceforge_downloads")).expanduser()
TIMEOUT = int(os.environ.get("VOICEFORGE_TIMEOUT", "3600"))  # scene renders can take minutes
MAX_RESULT_CHARS = int(os.environ.get("VOICEFORGE_MAX_RESULT_CHARS", "60000"))
PROTOCOL_VERSION = "2025-06-18"

INSTRUCTIONS = (
    "VoiceForge Studio: local expressive TTS, voice design, voice cloning and multi-character scene rendering. "
    "Start with `llm_guide` (prompting rules + script syntax) or `capabilities` (map of every operation). "
    "Quickest path from a script to audio: `cast_produce` (pass dry_run=true to preview the casting). "
    "Voices live in the library (`list_voices`); Clone Studio profiles (`list_profiles`) hold one clip per emotion "
    "and are picked per line automatically. After a render, use `check_scene`, `reroll_scene_line` and `use_take` "
    "to fix lines, then `export_scene_zip`."
)


# ----------------------------------------------------------------------------- HTTP

def _http(method: str, path: str, query: Optional[dict] = None, body: Optional[bytes] = None,
          content_type: Optional[str] = None) -> Tuple[int, str, bytes]:
    url = BASE_URL + path
    if query:
        url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
    headers = {"Content-Type": content_type} if content_type else {}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def _multipart(fields: Dict[str, Any], files: Dict[str, Path]) -> Tuple[bytes, str]:
    boundary = "----voiceforge" + uuid.uuid4().hex
    out = bytearray()
    for k, v in fields.items():
        if v is None:
            continue
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, p in files.items():
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{p.name}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n").encode()
        out += p.read_bytes() + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


# ----------------------------------------------------------------------------- OpenAPI -> tools

NAME_OVERRIDES = {
    # summary-derived names that read badly or collide
    ("get", "/api/llm-guide"): "llm_guide",
    ("get", "/api/scenes/{scene_id}/export.zip"): "export_scene_zip",
    ("get", "/api/scenes/{scene_id}/subtitles.srt"): "scene_subtitles_srt",
    ("get", "/api/scenes/{scene_id}/manifest.json"): "scene_manifest",
    ("post", "/v1/audio/speech"): "openai_audio_speech",
}


def _tool_name(method: str, path: str, op: dict, taken: set) -> str:
    name = NAME_OVERRIDES.get((method, path))
    if not name:
        summary = op.get("summary") or op.get("operationId") or f"{method} {path}"
        name = re.sub(r"[^a-z0-9]+", "_", summary.lower()).strip("_")
        name = re.sub(r"^api_", "", name)
    base, n = name, 2
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    taken.add(name)
    return name


def _resolve(schema: Any, spec: dict, depth: int = 0) -> Any:
    """Inline $refs (and simplify anyOf-with-null) so MCP clients get self-contained schemas."""
    if depth > 12:
        return {}
    if isinstance(schema, list):
        return [_resolve(s, spec, depth + 1) for s in schema]
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        target: Any = spec
        for part in schema["$ref"].lstrip("#/").split("/"):
            target = target.get(part, {})
        merged = {**target, **{k: v for k, v in schema.items() if k != "$ref"}}
        return _resolve(merged, spec, depth + 1)
    out = {k: _resolve(v, spec, depth + 1) for k, v in schema.items()}
    any_of = out.get("anyOf")
    if isinstance(any_of, list):
        non_null = [s for s in any_of if s.get("type") != "null"]
        if len(non_null) == 1 and len(any_of) == 2:
            out.pop("anyOf")
            out = {**non_null[0], **out}
    out.pop("title", None)
    return out


def build_tools(spec: dict) -> List[dict]:
    tools, taken = [], set()
    for path, methods in spec.get("paths", {}).items():
        for method, op in methods.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            props: Dict[str, Any] = {}
            required: List[str] = []
            path_params, query_params, body_fields, file_fields = [], [], [], []
            body_kind = None
            for prm in op.get("parameters", []):
                prm = _resolve(prm, spec)
                sch = dict(prm.get("schema", {}))
                if prm.get("description"):
                    sch["description"] = prm["description"]
                props[prm["name"]] = sch
                (path_params if prm["in"] == "path" else query_params).append(prm["name"])
                if prm.get("required"):
                    required.append(prm["name"])
            content = (op.get("requestBody") or {}).get("content", {})
            body_required = (op.get("requestBody") or {}).get("required", False)
            for ctype, kind in (("application/json", "json"), ("multipart/form-data", "multipart"),
                                ("application/x-www-form-urlencoded", "multipart")):
                if ctype in content:
                    body_kind = kind
                    sch = _resolve(content[ctype].get("schema", {}), spec)
                    for fname, fs in (sch.get("properties") or {}).items():
                        is_file = fs.get("format") == "binary" or fs.get("contentMediaType") or \
                            (fs.get("type") == "array" and (fs.get("items") or {}).get("format") == "binary")
                        key = f"{fname}_path" if is_file else fname
                        if key in props:
                            key = f"body_{key}"
                        if is_file:
                            props[key] = {"type": "string", "description": f"Local path of the file to upload as `{fname}`. "
                                                                            + fs.get("description", "")}
                            file_fields.append((key, fname))
                        else:
                            props[key] = fs
                            body_fields.append((key, fname))
                        if body_required and fname in (sch.get("required") or []):
                            required.append(key)
                    break
            doc = (op.get("description") or op.get("summary") or "").strip()
            doc = re.sub(r"\n{3,}", "\n\n", doc)
            tag = (op.get("tags") or ["Other"])[0]
            name = _tool_name(method, path, op, taken)
            tools.append({
                "name": name,
                "description": f"[{tag}] {method.upper()} {path}\n{doc}"[:3000],
                "inputSchema": {"type": "object", "properties": props, **({"required": required} if required else {})},
                "_route": {"method": method.upper(), "path": path, "path_params": path_params, "query_params": query_params,
                           "body_kind": body_kind, "body_fields": body_fields, "file_fields": file_fields},
            })
    return tools


_TOOLS_CACHE: Dict[str, Any] = {"at": 0.0, "tools": None}


def get_tools(refresh: bool = False) -> List[dict]:
    if refresh or _TOOLS_CACHE["tools"] is None or time.time() - _TOOLS_CACHE["at"] > 300:
        status, _, body = _http("GET", "/openapi.json")
        if status != 200:
            raise RuntimeError(f"Studio returned HTTP {status} for /openapi.json")
        _TOOLS_CACHE.update(at=time.time(), tools=build_tools(json.loads(body)))
    return _TOOLS_CACHE["tools"]


# ----------------------------------------------------------------------------- calling

def _save_binary(path: str, ctype: str, data: bytes) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.strip("/").replace("api/", ""))[:80] or "download"
    ext = mimetypes.guess_extension(ctype.split(";")[0].strip()) or ""
    if ext and not stem.endswith(ext):
        stem += ext
    dest = OUT_DIR / f"{int(time.time())}_{stem}"
    dest.write_bytes(data)
    return str(dest.resolve())


def call_tool(name: str, args: Optional[dict] = None) -> Tuple[bool, str]:
    """Returns (ok, text)."""
    args = dict(args or {})
    tool = next((t for t in get_tools() if t["name"] == name), None)
    if tool is None:
        tool = next((t for t in get_tools(refresh=True) if t["name"] == name), None)
    if tool is None:
        return False, f"Unknown tool '{name}'. Use `list` (CLI) or tools/list (MCP) to see the available tools."
    r = tool["_route"]
    path = r["path"]
    for p in r["path_params"]:
        if p not in args:
            return False, f"Missing required path parameter '{p}'."
        path = path.replace("{" + p + "}", urllib.parse.quote(str(args.pop(p)), safe=""))
    query = {p: args.pop(p) for p in r["query_params"] if p in args}
    query = {k: (str(v).lower() if isinstance(v, bool) else v) for k, v in query.items()}
    body, ctype = None, None
    if r["body_kind"] == "json":
        payload = {field: args.pop(key) for key, field in r["body_fields"] if key in args}
        payload.update(args)  # pass through anything extra rather than silently dropping it
        args = {}
        body, ctype = json.dumps(payload).encode(), "application/json"
    elif r["body_kind"] == "multipart":
        fields = {field: args.pop(key) for key, field in r["body_fields"] if key in args}
        files = {}
        for key, field in r["file_fields"]:
            if key in args:
                fp = Path(str(args.pop(key))).expanduser()
                if not fp.is_file():
                    return False, f"File not found: {fp}"
                files[field] = fp
        body, ctype = _multipart(fields, files)
    status, rtype, data = _http(r["method"], path, query or None, body, ctype)
    ok = 200 <= status < 300
    if "json" in rtype:
        text = data.decode("utf-8", errors="replace")
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=1)
        except ValueError:
            pass
    elif rtype.startswith("text/") and not rtype.startswith("text/html"):
        text = data.decode("utf-8", errors="replace")
    elif ok:
        text = json.dumps({"saved_to": _save_binary(path, rtype, data), "content_type": rtype, "bytes": len(data)})
    else:
        text = data.decode("utf-8", errors="replace")
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + f"\n… (truncated, {len(text)} chars total)"
    return ok, (text if ok else f"HTTP {status}: {text}")


# ----------------------------------------------------------------------------- MCP stdio server

def _send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _public(tool: dict) -> dict:
    return {k: v for k, v in tool.items() if not k.startswith("_")}


def serve() -> None:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if mid is None:  # notification
            continue
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "voiceforge", "version": "1.0"},
                    "instructions": INSTRUCTIONS,
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                try:
                    result = {"tools": [_public(t) for t in get_tools()]}
                except Exception as e:
                    result = {"tools": [{
                        "name": "studio_offline",
                        "description": f"VoiceForge Studio is not reachable at {BASE_URL} ({e}). Start it with ./start_studio.sh, "
                                       "then call this tool (or reconnect) to load the real tools.",
                        "inputSchema": {"type": "object", "properties": {}},
                    }]}
            elif method == "tools/call":
                name = params.get("name", "")
                if name == "studio_offline":
                    try:
                        n = len(get_tools(refresh=True))
                        ok, text = True, f"Studio is up with {n} tools. Reconnect the MCP server to load them."
                    except Exception as e:
                        ok, text = False, f"Still offline: {e}"
                else:
                    try:
                        ok, text = call_tool(name, params.get("arguments") or {})
                    except Exception as e:
                        ok, text = False, f"{type(e).__name__}: {e}"
                result = {"content": [{"type": "text", "text": text}], "isError": not ok}
            else:
                _send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}})
                continue
            _send({"jsonrpc": "2.0", "id": mid, "result": result})
        except Exception as e:
            _send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)}})


# ----------------------------------------------------------------------------- CLI

def main(argv: List[str]) -> int:
    if not argv or argv[0] in ("serve", "mcp"):
        serve()
        return 0
    cmd = argv[0]
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if cmd == "list":
        groups: Dict[str, List[dict]] = {}
        for t in get_tools():
            groups.setdefault(t["description"].split("]")[0].lstrip("["), []).append(t)
        for g, ts in groups.items():
            print(f"\n## {g}")
            for t in ts:
                first = t["description"].split("\n", 2)
                summary = first[1] if len(first) > 1 else ""
                print(f"  {t['name']:<34} {t['_route']['method']:<6} {t['_route']['path']}  {summary[:90]}")
        return 0
    if cmd == "describe" and len(argv) > 1:
        tool = next((t for t in get_tools() if t["name"] == argv[1]), None)
        if not tool:
            print(f"Unknown tool '{argv[1]}'", file=sys.stderr)
            return 2
        print(tool["description"])
        print(json.dumps(tool["inputSchema"], indent=1))
        return 0
    if cmd == "call" and len(argv) > 1:
        raw = argv[2] if len(argv) > 2 else ("{}" if sys.stdin.isatty() else sys.stdin.read() or "{}")
        ok, text = call_tool(argv[1], json.loads(raw))
        print(text)
        return 0 if ok else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
