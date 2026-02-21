# ShelterMed Voice Router App

A non-technical web app for field medical shelters that:
- uses your **actual hackathon router** (`generate_hybrid`) to choose tool calls,
- uses **Cactus transcription** for voice commands,
- supports **recording-only voice input** (no WAV upload UI),
- executes multiple actions and gives clear success/failure feedback.

## What this app demonstrates

1. Hybrid router in the loop (local/cloud decision logic from your real router file).
2. Voice-to-action flow using `cactus_transcribe`.
3. Fast-path routing for common medical commands (lower latency).
4. Argument repair for better tool-call accuracy.
5. Multi-action execution in one command.
6. User-friendly interface for medical staff (minimal technical exposure).

## Quick start with `uv`

From repo root:

```bash
cd apps/sheltermed_voice_router
uv sync
uv run python -m sheltermed_app.server
```

Open:

`http://127.0.0.1:9010`

## Router integration

By default, the server loads:

`sheltermed_app/cactus_general_router.py`

This router is app-specific and designed to generalize:
- semantic tool preselection (schema-driven),
- Cactus FunctionGemma tool-calling over selected tools,
- argument sanitization + type coercion,
- semantic fallback when local generation fails.

You can force a specific router path with:

```bash
uv run python -m sheltermed_app.server \
  --router-path "apps/sheltermed_voice_router/sheltermed_app/medical_router.py"
```

## Cactus transcription setup

This app expects Cactus to be built on your machine (same as hackathon setup).

If not yet installed:

```bash
git clone https://github.com/cactus-compute/cactus
cd cactus
source ./setup
cactus build --python
cactus download openai/whisper-small --reconvert
cd ..
```

Then run this app from repo root and pass whisper model path if needed:

```bash
uv run python -m sheltermed_app.server \
  --whisper-model "cactus/weights/whisper-small"
```

If Cactus is unavailable, text mode still works and UI shows voice status clearly.

## API endpoints

- `GET /api/health`
- `POST /api/text-command`
- `POST /api/voice-command` (expects `audio_b64`; used by in-browser recording flow)
- `GET /api/state` (latest shelter action state)

## Safety note

This is a hackathon prototype for decision support and operations flow, not a certified medical device.
