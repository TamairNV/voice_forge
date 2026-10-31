# VoiceForge Studio for AI agents

Everything the web pages do (`/` studio, `/cast` Casting Desk, `/voices` Voice Library, `/clone` Clone Studio)
goes through the HTTP API at `http://127.0.0.1:7860`, so an agent can do all of it. Start the studio with
`./start_studio.sh` first.

## Three ways in

| Way | Use it when | How |
| :-- | :-- | :-- |
| **MCP server** | You're an MCP client (Claude Code, Claude Desktop, …) | `claude mcp add voiceforge -- python3 /Users/tamair/voice_forge/voiceforge_mcp.py` |
| **CLI** | You can run shell commands | `python3 voiceforge_mcp.py list` · `describe <tool>` · `call <tool> '<json>'` |
| **HTTP / Python** | Anything else | `GET /api/capabilities`, `/openapi.json`, or `voiceforge_client.py` (stdlib only) |

The MCP server and CLI build one tool per API operation from the live `/openapi.json`, so they never miss an
endpoint. Arguments are one flat JSON object: path params, query params and body fields together. Uploads take a
local path in `file_path`. Audio, zip and srt downloads are saved to `$VOICEFORGE_OUT` (default
`./voiceforge_downloads`) and the tool returns the path.

## Discovery

- `GET /api/capabilities`: every operation grouped by area, built from the routes.
- `GET /api/llm-guide`: prompting rules (voice descriptions, Maya-1 tags, cloning rules) and script syntax.
- `GET /openapi.json` or `/docs`: full schemas.

## Common jobs

**Script to finished scene, one call** (what the Casting Desk does):

```bash
python3 voiceforge_mcp.py call cast_produce '{
  "script": "Tamair (panicking): Where are my keys?!\nBob (deadpan): In your hand.\nTamair: ...oh.",
  "voices": {"Bob": "dry, bored London bloke in his 40s, low flat voice"},
  "dry_run": true
}'
```

`voices` maps a character to a library voice (id or name), `"profile:<name>"` for a Clone Studio profile, or a
plain-English description to design a new voice. Anyone unlisted gets a same-named profile, then the suggested
library voice, then a new voice from the script. `dry_run` returns `scene_request` (a `/api/scenes/render`
payload) without rendering. Drop it, or post that payload to `render_scene`, to render.

**Render with full control:** `POST /api/scenes/render` with `cast` + `lines` or `script_text`. Per line:
`emotion`, `direction`, `blend_with` / `blend_strength`, `context_reset`, `continuity_group`, `pause_after_ms`
(negative = overlap), `temperature`, `seed`. Scene-wide: `takes_per_line` (best of N), `natural_pacing`,
`context_continuity`, `temperature`.

**Clone Studio profiles (one person, one clip per emotion):**

- Cast a speaker as `{"profile": "Tamair"}` or `{"profile_id": "prof_..."}`, or just name them exactly like
  the profile. Each line then uses the closest emotion clip, picked from `emotion`, else `direction`
  (`Tamair (furious): ...` in `script_text`), else the text (ALL CAPS = shouting). Set `use_profiles: false`
  to turn this off.
- Build one: `create_profile` → `add_profile_clip_from_path` (`emotion`, `audio_path`, optional `script`) or
  `add_profile_clip` (upload) or `link_profile_clip` (existing voice). `profile_emotions` lists the slots,
  acting notes and suggested lines; `pick_profile_clip` previews which clip a line gets.

**Fix a rendered scene:** `check_scene` (Whisper-checks every line, returns `flagged`) → `reroll_scene_line`
(new seed, text, speed, blend…) or `use_scene_take` (switch to a runner-up from `candidates`) →
`export_scene_zip` / `scene_subtitles_srt` / `scene_manifest`.

**Voice Library:** `list_voices`, `update_character` (rename, `ref_text`, `tags`, `favorite`),
`retranscribe_character`, `delete_character`. New voices: `design_and_lock_character`, `synthesize` then
`save_character_from_take` (audition and keep the best), `create_character_from_recording` (local path),
`upload_character`.

**Saved casts:** `save_cast`, `list_casts`, `render_saved_cast` (id or name; re-renders with the same voices,
profiles, variants and blends), `delete_cast`.

## Things to know

- Only one model is in memory at a time; mixing engines in a scene costs swaps. Calls are serialised.
- Renders take a while (the clone model loads on the first line). The MCP/CLI timeout is an hour
  (`VOICEFORGE_TIMEOUT`).
- The Claude-backed steps (`cast_extract`, `cast_improve_description`, `generate_recording_prompt`) need the
  server's `claude` CLI signed in (`cast_claude_status`). `cast_extract` / `cast_produce` fall back to the
  `Name: line` parser if it isn't (or pass `use_claude: false`).
- Mic recording in the browser has no API twin; record or find a file, then use `add_profile_clip_from_path`.
