from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sheltermed_app import cactus_general_router
from sheltermed_app.server import AppContext
from sheltermed_app.tool_schemas import TOOL_DEFINITIONS
from sheltermed_app.transcription_service import CactusTranscriber


def test_fastpath_routes_inventory_and_restock_quickly() -> None:
    query = "Check oxygen cylinder inventory and request urgent restock of 15."
    t0 = time.perf_counter()
    out = cactus_general_router.generate_hybrid(
        messages=[{"role": "user", "content": query}],
        tools=TOOL_DEFINITIONS,
    )
    elapsed = time.perf_counter() - t0
    assert out["source"] in {"cactus-general-fastpath", "cactus-general", "semantic-fallback"}
    names = [c["name"] for c in out["function_calls"]]
    assert "check_inventory" in names
    assert "request_restock" in names
    # Fast path should stay very fast on CPU.
    assert elapsed < 0.08


def test_fastpath_cache_hit_on_repeated_query() -> None:
    query = "Check oxygen cylinder inventory and request urgent restock of 15."
    cactus_general_router.generate_hybrid(
        messages=[{"role": "user", "content": query}],
        tools=TOOL_DEFINITIONS,
    )
    out2 = cactus_general_router.generate_hybrid(
        messages=[{"role": "user", "content": query}],
        tools=TOOL_DEFINITIONS,
    )
    assert out2.get("router_profile", {}).get("cache_hit") is True


def test_transcriber_rejects_none_like_payloads() -> None:
    tr = CactusTranscriber.__new__(CactusTranscriber)
    tr._error = ""
    tr._model = object()
    tr._json = json
    tr._cactus_init = None
    tr._cactus_destroy = None
    tr._prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    tr._ensure_model = lambda: None

    responses = iter(
        [
            '{"response": null}',
            '{"response": "None"}',
        ]
    )
    tr._cactus_transcribe = lambda *_args, **_kwargs: next(responses)

    try:
        tr.transcribe_wav_bytes(b"RIFFxxxxWAVEfmt ")
        assert False, "Expected RuntimeError for invalid/None transcription"
    except RuntimeError as exc:
        assert "empty response" in str(exc).lower()


@dataclass
class _FakeStatus:
    ready: bool
    whisper_model_path: str
    error: str


class _FakeTranscriber:
    def __init__(self, values: list[str]) -> None:
        self._values = values
        self._i = 0

    def status(self) -> _FakeStatus:
        return _FakeStatus(ready=True, whisper_model_path="x", error="")

    def transcribe_base64_wav(self, _audio: str) -> str:
        v = self._values[self._i]
        self._i += 1
        return v


def test_repeated_voice_processing_consistent() -> None:
    fake_ctx = AppContext.__new__(AppContext)
    fake_ctx.transcriber = _FakeTranscriber(["Check inventory", "Notify team"])
    fake_ctx.process_command = lambda text, input_mode: (
        200,
        {"ok": True, "input_mode": input_mode, "command_text": text, "summary": "ok", "outcomes": [], "state": {}},
    )

    code1, payload1 = AppContext.process_voice_audio(fake_ctx, "Zm9v")
    code2, payload2 = AppContext.process_voice_audio(fake_ctx, "YmFy")

    assert code1 == 200
    assert code2 == 200
    assert payload1["transcript"] == "Check inventory"
    assert payload2["transcript"] == "Notify team"


def test_voice_rejects_none_transcript() -> None:
    fake_ctx = AppContext.__new__(AppContext)
    fake_ctx.transcriber = _FakeTranscriber(["None"])
    fake_ctx.process_command = lambda text, input_mode: (200, {"ok": True, "command_text": text, "input_mode": input_mode})
    code, payload = AppContext.process_voice_audio(fake_ctx, "Zm9v")
    assert code == 400
    assert "unclear" in payload["error"].lower()
