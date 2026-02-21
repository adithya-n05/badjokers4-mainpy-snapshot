from __future__ import annotations

import argparse
import json
import mimetypes
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from .router_adapter import RouterAdapter, find_repo_root
from .tool_executor import ActionExecutor
from .tool_schemas import TOOL_DEFINITIONS
from .transcription_service import CactusTranscriber


def read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Payload must be a JSON object")
    return data


def write_json(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class AppContext:
    def __init__(self, repo_root: Path, router_path: Path, whisper_model_path: str, reload_router: bool) -> None:
        self.repo_root = repo_root
        self.router = RouterAdapter(router_path=router_path, repo_root=repo_root, reload_on_change=reload_router)
        self.transcriber = CactusTranscriber(whisper_model_path=whisper_model_path)
        self.executor = ActionExecutor()
        self.tools = TOOL_DEFINITIONS
        self.static_root = Path(__file__).resolve().parent / "static"

    def close(self) -> None:
        self.transcriber.close()

    def health_payload(self) -> Dict[str, Any]:
        rs = self.router.status()
        ts = self.transcriber.status()
        return {
            "ok": True,
            "router": asdict(rs),
            "transcription": asdict(ts),
            "tool_count": len(self.tools),
            "state": self.executor.state.snapshot(),
        }

    def _fallback_calls(self, text: str) -> List[Dict[str, Any]]:
        t = text.lower()
        calls: List[Dict[str, Any]] = []
        if "restock" in t or "stock" in t or "inventory" in t:
            calls.append({"name": "check_inventory", "arguments": {"item": "oxygen cylinder"}})
        if "refer" in t or "hospital" in t:
            calls.append(
                {
                    "name": "create_referral",
                    "arguments": {
                        "patient_name": "Unspecified Patient",
                        "destination_facility": "Regional Hospital",
                        "reason": "Needs higher-level evaluation",
                        "urgency": "urgent",
                    },
                }
            )
        if "alert" in t or "broadcast" in t:
            calls.append(
                {
                    "name": "broadcast_alert",
                    "arguments": {"alert_level": "urgent", "message": text},
                }
            )
        if not calls:
            calls.append(
                {
                    "name": "notify_team",
                    "arguments": {
                        "recipient": "Medical Team Lead",
                        "message": text,
                        "channel": "internal",
                    },
                }
            )
        return calls

    def process_command(self, text: str, input_mode: str) -> Tuple[int, Dict[str, Any]]:
        text = " ".join((text or "").split())
        if not text:
            return 400, {"ok": False, "error": "No command text provided."}

        t0 = time.perf_counter()
        route_response = self.router.route(text, self.tools)
        route_latency = (time.perf_counter() - t0) * 1000.0

        if route_response["ok"]:
            routed = route_response["result"]
            function_calls = routed.get("function_calls", [])
            route_source = routed.get("source", "router")
            used_fallback = False
            router_error = ""
        else:
            routed = {}
            function_calls = []
            route_source = "router_error"
            used_fallback = True
            router_error = route_response["error"]

        if not isinstance(function_calls, list) or not function_calls:
            function_calls = self._fallback_calls(text)
            used_fallback = True

        outcomes = self.executor.execute_many(function_calls)
        success_count = sum(1 for o in outcomes if o["success"])
        failure_count = len(outcomes) - success_count
        done_msgs = [o["message"] for o in outcomes if o["success"]]
        fail_msgs = [o["message"] for o in outcomes if not o["success"]]

        summary = []
        if done_msgs:
            summary.append(f"{success_count} action(s) completed.")
        if fail_msgs:
            summary.append(f"{failure_count} action(s) failed and need review.")
        if used_fallback:
            summary.append("Used safeguard routing fallback.")

        payload = {
            "ok": True,
            "input_mode": input_mode,
            "command_text": text,
            "router": {
                "source": route_source,
                "latency_ms": route_latency,
                "router_total_time_ms": routed.get("total_time_ms", 0.0),
                "profile": routed.get("router_profile", {}),
                "used_fallback": used_fallback,
                "error": router_error,
            },
            "function_calls": function_calls,
            "outcomes": outcomes,
            "summary": " ".join(summary),
            "state": self.executor.state.snapshot(),
        }
        return 200, payload

    def process_voice_audio(self, audio_b64: str) -> Tuple[int, Dict[str, Any]]:
        ts = self.transcriber.status()
        if not ts.ready:
            return 400, {"ok": False, "error": ts.error}

        t0 = time.perf_counter()
        transcript = self.transcriber.transcribe_base64_wav(audio_b64)
        transcribe_ms = (time.perf_counter() - t0) * 1000.0
        code, payload = self.process_command(transcript, input_mode="voice")
        if code == 200:
            payload["transcript"] = transcript
            payload["transcription_ms"] = transcribe_ms
        return code, payload


class ShelterHandler(BaseHTTPRequestHandler):
    server: "ShelterServer"

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/health":
            write_json(self, 200, self.server.ctx.health_payload())
            return
        if route == "/api/state":
            write_json(self, 200, {"ok": True, "state": self.server.ctx.executor.state.snapshot()})
            return
        if route in {"/", "/index.html", "/styles.css", "/app.js"}:
            self._serve_asset(route)
            return
        write_json(self, 404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            data = read_json(self)
            if route == "/api/text-command":
                text = str(data.get("text", "")).strip()
                code, payload = self.server.ctx.process_command(text, input_mode="text")
                write_json(self, code, payload)
                return
            if route == "/api/voice-command":
                audio_b64 = str(data.get("audio_b64", "")).strip()
                if not audio_b64:
                    write_json(self, 400, {"ok": False, "error": "audio_b64 is required"})
                    return
                code, payload = self.server.ctx.process_voice_audio(audio_b64)
                write_json(self, code, payload)
                return
            write_json(self, 404, {"ok": False, "error": "not_found"})
        except Exception as exc:
            write_json(self, 400, {"ok": False, "error": str(exc)})

    def _serve_asset(self, route: str) -> None:
        if route in {"/", "/index.html"}:
            filename = "index.html"
        else:
            filename = route.lstrip("/")

        asset = self.server.ctx.static_root / filename
        if not asset.exists():
            write_json(self, 404, {"ok": False, "error": f"missing asset {filename}"})
            return

        body = asset.read_bytes()
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


class ShelterServer(ThreadingHTTPServer):
    def __init__(self, addr: Tuple[str, int], ctx: AppContext) -> None:
        super().__init__(addr, ShelterHandler)
        self.ctx = ctx


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    default_router = here.parent / "cactus_general_router.py"
    whisper_candidates = [
        repo_root / "cactus" / "weights" / "whisper-small",
        repo_root.parent / "Cactus-Deepmind-Hackathon-Prep-26" / "cactus" / "weights" / "whisper-small",
    ]
    default_whisper = str(next((p for p in whisper_candidates if p.exists()), whisper_candidates[0]))

    p = argparse.ArgumentParser(description="ShelterMed web app using actual router + Cactus transcription")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9010)
    p.add_argument("--router-path", default=str(default_router))
    p.add_argument("--whisper-model", default=default_whisper)
    p.add_argument("--no-router-reload", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path.cwd())
    router_path = Path(args.router_path).resolve()
    ctx = AppContext(
        repo_root=repo_root,
        router_path=router_path,
        whisper_model_path=args.whisper_model,
        reload_router=not args.no_router_reload,
    )

    health = ctx.health_payload()
    print(f"[sheltermed] starting http://{args.host}:{args.port}")
    print(f"[sheltermed] router ready={health['router']['ready']} path={health['router']['router_path']}")
    if not health["router"]["ready"]:
        print(f"[sheltermed] router error={health['router']['error']}")
    print(f"[sheltermed] cactus transcription ready={health['transcription']['ready']}")
    if not health["transcription"]["ready"]:
        print(f"[sheltermed] transcription error={health['transcription']['error']}")

    server = ShelterServer((args.host, args.port), ctx)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ctx.close()
        server.server_close()


if __name__ == "__main__":
    main()
