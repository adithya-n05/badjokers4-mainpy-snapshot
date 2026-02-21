from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / ".git").exists():
            return parent
    return start.resolve()


@dataclass
class RouterStatus:
    ready: bool
    router_path: str
    error: str
    loaded_at: float


class RouterAdapter:
    def __init__(self, router_path: Path, repo_root: Path, reload_on_change: bool = True) -> None:
        self.router_path = router_path
        self.repo_root = repo_root
        self.reload_on_change = reload_on_change
        self._route_fn: Optional[Callable[..., Dict[str, Any]]] = None
        self._error = ""
        self._loaded_at = 0.0
        self._last_mtime = 0.0
        self._load_router()

    def _prepare_paths(self) -> None:
        cactus_candidates = [
            self.repo_root / "cactus" / "python" / "src",
            self.repo_root.parent / "Cactus-Deepmind-Hackathon-Prep-26" / "cactus" / "python" / "src",
        ]
        for cactus_src in cactus_candidates:
            if cactus_src.exists():
                p = str(cactus_src)
                if p not in sys.path:
                    sys.path.insert(0, p)

        repo_str = str(self.repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def _load_router(self) -> None:
        self._prepare_paths()
        self._route_fn = None
        self._error = ""
        self._loaded_at = 0.0

        if not self.router_path.exists():
            self._error = f"Router file not found: {self.router_path}"
            return

        try:
            mtime = self.router_path.stat().st_mtime
            module_name = f"shelter_router_{int(time.time() * 1000)}"
            spec = importlib.util.spec_from_file_location(module_name, str(self.router_path))
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to build module spec")
            module = importlib.util.module_from_spec(spec)

            old_cwd = Path.cwd()
            try:
                os.chdir(self.repo_root)
                spec.loader.exec_module(module)
            finally:
                os.chdir(old_cwd)

            # The hackathon router uses a repo-relative FunctionGemma weight path.
            # Canonicalize to an absolute path so calls work regardless of server cwd.
            fg_path = getattr(module, "functiongemma_path", None)
            if isinstance(fg_path, str) and fg_path and not Path(fg_path).is_absolute():
                module.functiongemma_path = str((self.repo_root / fg_path).resolve())

            route_fn = getattr(module, "generate_hybrid", None)
            if not callable(route_fn):
                raise RuntimeError("Router module has no callable `generate_hybrid`")

            self._route_fn = route_fn
            self._last_mtime = mtime
            self._loaded_at = time.time()
        except Exception as exc:
            self._error = str(exc)

    def _ensure_loaded(self) -> None:
        if self.reload_on_change and self.router_path.exists():
            mtime = self.router_path.stat().st_mtime
            if mtime > self._last_mtime:
                self._load_router()
        if self._route_fn is None and not self._error:
            self._load_router()

    def status(self) -> RouterStatus:
        self._ensure_loaded()
        return RouterStatus(
            ready=self._route_fn is not None,
            router_path=str(self.router_path),
            error=self._error,
            loaded_at=self._loaded_at,
        )

    def route(self, user_text: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._ensure_loaded()
        if self._route_fn is None:
            return {"ok": False, "error": self._error or "Router unavailable", "result": {}}

        messages = [{"role": "user", "content": user_text}]
        try:
            result = self._route_fn(messages, tools)
            if not isinstance(result, dict):
                result = {}
            result.setdefault("function_calls", [])
            result.setdefault("total_time_ms", 0.0)
            result.setdefault("source", "router")
            return {"ok": True, "error": "", "result": result}
        except Exception as exc:
            return {"ok": False, "error": f"Router call failed: {exc}", "result": {}}
