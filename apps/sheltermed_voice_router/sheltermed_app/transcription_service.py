from __future__ import annotations

import base64
import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TranscriptionStatus:
    ready: bool
    whisper_model_path: str
    error: str


class CactusTranscriber:
    def __init__(self, whisper_model_path: str) -> None:
        self.whisper_model_path = whisper_model_path
        self._error = ""
        self._model = None
        self._json = None
        self._cactus_init = None
        self._cactus_destroy = None
        self._cactus_transcribe = None
        self._prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
        self._last_audio_hash = ""
        self._last_transcript = ""
        self._setup()

    def _setup(self) -> None:
        repo_root = Path(__file__).resolve()
        roots = [repo_root]
        roots.extend(repo_root.parents)
        git_root = None
        for root in roots:
            if (root / ".git").exists():
                git_root = root
                break
        if git_root is None:
            git_root = Path.cwd()

        cactus_candidates = [
            git_root / "cactus" / "python" / "src",
            git_root.parent / "Cactus-Deepmind-Hackathon-Prep-26" / "cactus" / "python" / "src",
        ]
        for candidate in cactus_candidates:
            if candidate.exists():
                cpath = str(candidate)
                if cpath not in sys.path:
                    sys.path.insert(0, cpath)

        try:
            import json as _json
            from cactus import cactus_destroy, cactus_init, cactus_transcribe

            self._json = _json
            self._cactus_init = cactus_init
            self._cactus_destroy = cactus_destroy
            self._cactus_transcribe = cactus_transcribe
        except Exception as exc:
            self._error = (
                "Cactus import failed. Ensure Cactus is built and Python bindings are available. "
                f"Underlying error: {exc}"
            )
            return

        if not Path(self.whisper_model_path).exists():
            self._error = f"Whisper model path not found: {self.whisper_model_path}"

    def status(self) -> TranscriptionStatus:
        return TranscriptionStatus(
            ready=not self._error,
            whisper_model_path=self.whisper_model_path,
            error=self._error,
        )

    def _ensure_model(self) -> None:
        if self._error:
            raise RuntimeError(self._error)
        if self._model is None:
            assert self._cactus_init is not None
            try:
                self._model = self._cactus_init(self.whisper_model_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to initialize Cactus Whisper model at {self.whisper_model_path}: {exc}"
                ) from exc

    def _reset_model(self) -> None:
        if self._model is not None and self._cactus_destroy is not None:
            try:
                self._cactus_destroy(self._model)
            except Exception:
                pass
            finally:
                self._model = None

    def _run_transcription(self, wav_path: str) -> str:
        assert self._cactus_transcribe is not None
        assert self._json is not None
        prompts = [
            self._prompt,
            "<|startoftranscript|><|en|><|transcribe|>",
        ]
        for prompt in prompts:
            try:
                raw = self._cactus_transcribe(self._model, wav_path, prompt=prompt)
                parsed = self._json.loads(raw)
            except Exception:
                continue
            value = parsed.get("response")
            if value is None:
                value = parsed.get("text")
            if value is None:
                value = parsed.get("transcript")
            text = "" if value is None else str(value).strip()
            if text and text.lower() not in {"none", "null", "undefined"}:
                return text
        return ""

    def transcribe_base64_wav(self, audio_b64: str) -> str:
        try:
            audio_bytes = base64.b64decode(audio_b64, validate=True)
        except Exception as exc:
            raise RuntimeError(f"Invalid base64 audio payload: {exc}") from exc
        return self.transcribe_wav_bytes(audio_bytes)

    def transcribe_wav_bytes(self, audio_bytes: bytes) -> str:
        self._ensure_model()
        tmp_path = ""
        audio_hash = hashlib.sha1(audio_bytes).hexdigest()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            text = self._run_transcription(tmp_path)

            # Guard against stale decoder state: if audio changed but transcript
            # repeats exactly, force one fresh-model retry.
            if (
                text
                and self._last_audio_hash
                and self._last_transcript
                and audio_hash != self._last_audio_hash
                and text == self._last_transcript
            ):
                self._reset_model()
                self._ensure_model()
                retry_text = self._run_transcription(tmp_path)
                if retry_text:
                    text = retry_text

            if not text:
                raise RuntimeError("Transcription returned empty response")

            self._last_audio_hash = audio_hash
            self._last_transcript = text
            return text
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def close(self) -> None:
        self._reset_model()
