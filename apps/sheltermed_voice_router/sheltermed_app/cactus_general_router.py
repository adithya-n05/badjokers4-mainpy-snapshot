from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

functiongemma_path = "cactus/weights/functiongemma-270m-it"

_MODEL: Any = None
_MODEL_PATH = ""
_TOOL_CACHE: Dict[str, Dict[str, Any]] = {}
_CACTUS_INIT = None
_CACTUS_COMPLETE = None
_KNOWN_ITEMS = [
    "oxygen cylinder",
    "iv fluids",
    "bandages",
    "gloves",
    "paracetamol",
    "ibuprofen",
    "salbutamol",
    "antibiotics",
]
_KNOWN_MEDS = ["paracetamol", "ibuprofen", "cetirizine", "amoxicillin", "salbutamol"]


def _repo_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / ".git").exists():
            return parent
    return start.resolve()


_ROOT = _repo_root(Path(__file__).resolve())


def _candidate_repo_roots() -> List[Path]:
    roots = [_ROOT]
    sibling = _ROOT.parent / "Cactus-Deepmind-Hackathon-Prep-26"
    if sibling.exists():
        roots.append(sibling)
    return roots


def _ensure_cactus_import() -> None:
    global _CACTUS_INIT, _CACTUS_COMPLETE
    if _CACTUS_INIT is not None and _CACTUS_COMPLETE is not None:
        return

    cactus_srcs = []
    for root in _candidate_repo_roots():
        cactus_srcs.append(root / "cactus" / "python" / "src")
    for src in cactus_srcs:
        if src.exists():
            p = str(src)
            if p not in sys.path:
                sys.path.insert(0, p)

    from cactus import cactus_complete, cactus_init

    _CACTUS_INIT = cactus_init
    _CACTUS_COMPLETE = cactus_complete


def _resolve_functiongemma_path() -> str:
    override = os.environ.get("SHELTERMED_FUNCTIONGEMMA_PATH", "").strip()
    if override:
        p = Path(override)
        return str(p if p.is_absolute() else (_ROOT / p).resolve())

    fg = Path(functiongemma_path)
    if fg.is_absolute() and fg.exists():
        return str(fg)

    for root in _candidate_repo_roots():
        candidate = (root / functiongemma_path).resolve()
        if candidate.exists():
            return str(candidate)
    return str((_ROOT / functiongemma_path).resolve())


def _normalize(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _clauses(query: str) -> List[str]:
    parts = re.split(r"\b(?:and then|then|and|;|\.)\b", query, flags=re.IGNORECASE)
    out = [_normalize(p) for p in parts if _normalize(p)]
    return out or [query]


def _tool_signature(tools: List[Dict[str, Any]]) -> str:
    names = [str(t.get("name", "")) for t in tools if isinstance(t, dict)]
    return "|".join(sorted(names))


def _tool_text(tool: Dict[str, Any]) -> str:
    name = _normalize(tool.get("name"))
    desc = _normalize(tool.get("description"))
    params = tool.get("parameters", {})
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    fields = []
    if isinstance(props, dict):
        for key, spec in props.items():
            if isinstance(spec, dict):
                fields.append(f"{key} {_normalize(spec.get('description'))}")
            else:
                fields.append(str(key))
    return _normalize(" ".join([name, desc, " ".join(fields)]))


def _build_index(tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    sig = _tool_signature(tools)
    cached = _TOOL_CACHE.get(sig)
    if cached is not None:
        return cached

    entries: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        text = _tool_text(tool)
        tks = _tokens(text)
        entries.append(
            {
                "name": str(tool["name"]),
                "tool": tool,
                "text": text,
                "tokens": set(tks),
            }
        )
    idx = {"entries": entries}
    _TOOL_CACHE[sig] = idx
    return idx


def _score(clause: str, entry: Dict[str, Any]) -> float:
    c_tokens = _tokens(clause)
    if not c_tokens:
        return 0.0
    cset = set(c_tokens)
    overlap = len(cset & entry["tokens"])
    soft = sum(1 for t in c_tokens if t in entry["text"])
    return overlap * 1.8 + soft * 0.35


def _top_tools_for_clause(clause: str, index: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for entry in index["entries"]:
        scored.append((_score(clause, entry), entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e["tool"] for s, e in scored[:k] if s > 0]


def _required(tool: Dict[str, Any]) -> List[str]:
    params = tool.get("parameters", {})
    if isinstance(params, dict):
        req = params.get("required", [])
        if isinstance(req, list):
            return [str(x) for x in req]
    return []


def _props(tool: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    params = tool.get("parameters", {})
    if isinstance(params, dict):
        props = params.get("properties", {})
        if isinstance(props, dict):
            out: Dict[str, Dict[str, Any]] = {}
            for k, v in props.items():
                if isinstance(v, dict):
                    out[str(k)] = v
            return out
    return {}


def _extract_number(query: str, default: int = 1) -> int:
    m = re.search(r"\b(\d{1,4})\b", query)
    if m:
        return max(1, int(m.group(1)))
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "ten": 10,
        "fifteen": 15,
        "twenty": 20,
        "thirty": 30,
    }
    for tok in _tokens(query):
        if tok in words:
            return words[tok]
    return default


def _extract_name(query: str) -> str:
    m = re.search(r"\b(?:patient|for)\s+([A-Z][A-Za-z' -]{1,40}?)(?:\s+(?:with|in|at|to|and)\b|[,.;]|$)", query)
    if m:
        return _normalize(m.group(1))
    return "Unknown"


def _extract_zone(query: str) -> str:
    m = re.search(r"\bzone\s+([A-Za-z0-9-]+)\b", query, re.IGNORECASE)
    if m:
        return f"Zone {m.group(1).upper()}"
    return "intake"


def _find_item(query: str) -> str:
    q = query.lower()
    for item in _KNOWN_ITEMS:
        if item in q:
            return item
    if "oxygen" in q:
        return "oxygen cylinder"
    if "fluid" in q:
        return "iv fluids"
    return "oxygen cylinder"


def _find_med(query: str) -> str:
    q = query.lower()
    for med in _KNOWN_MEDS:
        if med in q:
            return med
    return "paracetamol"


def _find_priority(query: str) -> str:
    q = query.lower()
    if "urgent" in q or "critical" in q:
        return "urgent"
    if "high" in q:
        return "high"
    if "low" in q:
        return "low"
    return "medium"


def _find_triage(query: str) -> str:
    q = query.lower()
    for t in ["red", "orange", "yellow", "green"]:
        if t in q:
            return t
    return "yellow"


def _fallback_for_param(param: str, ptype: str, query: str) -> Any:
    pl = param.lower()
    q = _normalize(query)
    if ptype in {"integer", "number"}:
        return _extract_number(q, default=1)
    if ptype == "boolean":
        return True
    if "medication" in pl:
        return _find_med(q)
    if "triage" in pl:
        return _find_triage(q)
    if "priority" in pl:
        return _find_priority(q)
    if "name" in pl or "patient" in pl or "recipient" in pl:
        return _extract_name(q)
    if "zone" in pl:
        return _extract_zone(q)
    if "item" in pl:
        return _find_item(q)
    if "message" in pl:
        return q
    return q[:120]


def _coerce(value: Any, ptype: str, query: str, param: str) -> Any:
    if ptype == "string":
        return _normalize(value) if value is not None else _fallback_for_param(param, ptype, query)
    if ptype == "integer":
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        m = re.search(r"-?\d+", str(value or ""))
        return int(m.group(0)) if m else int(_fallback_for_param(param, ptype, query))
    if ptype == "number":
        if isinstance(value, (int, float)):
            return float(value)
        m = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
        return float(m.group(0)) if m else float(_fallback_for_param(param, ptype, query))
    if ptype == "boolean":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes", "y"}
    return value


def _sanitize_calls(calls: List[Dict[str, Any]], tools: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    tool_by_name = {str(t.get("name", "")): t for t in tools if isinstance(t, dict)}
    out: List[Dict[str, Any]] = []
    seen = set()
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = _normalize(call.get("name"))
        if not name or name not in tool_by_name:
            continue
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                args = parsed if isinstance(parsed, dict) else {}
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}

        tool = tool_by_name[name]
        props = _props(tool)
        req = _required(tool)
        cleaned: Dict[str, Any] = {}
        for key, spec in props.items():
            if key in args:
                ptype = str(spec.get("type", "string")).lower()
                cleaned[key] = _coerce(args.get(key), ptype, query, key)
        for key in req:
            if key not in cleaned or cleaned[key] in {"", None}:
                ptype = str(props.get(key, {}).get("type", "string")).lower()
                cleaned[key] = _fallback_for_param(key, ptype, query)

        sig = (name, tuple(sorted((k, str(v).lower()) for k, v in cleaned.items())))
        if sig in seen:
            continue
        seen.add(sig)
        out.append({"name": name, "arguments": cleaned})
    return out


def _ensure_model() -> Any:
    global _MODEL, _MODEL_PATH, _CACTUS_INIT
    _ensure_cactus_import()
    path = _resolve_functiongemma_path()
    if _MODEL is not None and _MODEL_PATH == path:
        return _MODEL
    assert _CACTUS_INIT is not None
    _MODEL = _CACTUS_INIT(path)
    _MODEL_PATH = path
    return _MODEL


def _call_cactus(query: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    global _CACTUS_COMPLETE
    model = _ensure_model()
    assert _CACTUS_COMPLETE is not None
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    raw = _CACTUS_COMPLETE(
        model,
        [{"role": "user", "content": query}],
        tools=cactus_tools,
        force_tools=bool(cactus_tools),
        tool_rag_top_k=min(4, len(cactus_tools)),
        temperature=0.0,
        max_tokens=196,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    parsed = json.loads(raw)
    calls = parsed.get("function_calls", [])
    if not isinstance(calls, list):
        calls = []
    return {
        "function_calls": calls,
        "total_time_ms": float(parsed.get("total_time_ms", 0.0) or 0.0),
    }


def _semantic_fallback(query: str, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    idx = _build_index(tools)
    all_calls: List[Dict[str, Any]] = []
    for clause in _clauses(query):
        ranked = _top_tools_for_clause(clause, idx, k=1)
        if not ranked:
            continue
        tool = ranked[0]
        req = _required(tool)
        props = _props(tool)
        args: Dict[str, Any] = {}
        for key in req:
            ptype = str(props.get(key, {}).get("type", "string")).lower()
            args[key] = _fallback_for_param(key, ptype, clause)
        all_calls.append({"name": str(tool.get("name", "")), "arguments": args})
    return _sanitize_calls(all_calls, tools, query)


def _tool_names(tools: List[Dict[str, Any]]) -> set[str]:
    return {str(t.get("name", "")) for t in tools if isinstance(t, dict)}


def _fastpath_calls(query: str, tool_names: set[str]) -> List[Dict[str, Any]]:
    q = _normalize(query)
    ql = q.lower()
    calls: List[Dict[str, Any]] = []
    reg_name: Optional[str] = None

    if "register" in ql and "patient" in ql and "register_patient" in tool_names:
        n = re.search(r"\bregister(?:\s+patient)?\s+([A-Za-z][A-Za-z' -]{1,40}?)(?:\s+(?:with|for|in|at)\b|[,.;]|$)", q, re.IGNORECASE)
        c = re.search(r"\bwith\s+(.+?)(?:\s+(?:in\s+zone|assign|set|notify|and|then)\b|[,.;]|$)", q, re.IGNORECASE)
        reg_name = _normalize(n.group(1)) if n else "Unknown"
        calls.append(
            {
                "name": "register_patient",
                "arguments": {
                    "patient_name": reg_name,
                    "main_complaint": _normalize(c.group(1)) if c else "unspecified complaint",
                    "location_zone": _extract_zone(q),
                },
            }
        )

    if ("inventory" in ql or "stock" in ql or "supply" in ql) and "check_inventory" in tool_names:
        calls.append({"name": "check_inventory", "arguments": {"item": _find_item(q)}})

    if ("restock" in ql or "resupply" in ql) and "request_restock" in tool_names:
        calls.append(
            {
                "name": "request_restock",
                "arguments": {
                    "item": _find_item(q),
                    "quantity": _extract_number(q, default=10),
                    "priority": _find_priority(q),
                },
            }
        )

    if "triage" in ql and "assign_triage_level" in tool_names:
        calls.append(
            {
                "name": "assign_triage_level",
                "arguments": {
                    "patient_name": reg_name or _extract_name(q),
                    "triage_level": _find_triage(q),
                    "reason": "Set from command",
                },
            }
        )

    if ("follow up" in ql or "follow-up" in ql or "reassess" in ql) and "set_followup_timer" in tool_names:
        calls.append(
            {
                "name": "set_followup_timer",
                "arguments": {
                    "patient_name": reg_name or _extract_name(q),
                    "minutes": _extract_number(q, default=20),
                    "task": "reassess",
                },
            }
        )

    if ("notify" in ql or "message" in ql) and "notify_team" in tool_names:
        r = re.search(r"\bnotify\s+(?:the\s+)?([A-Za-z][A-Za-z .'-]{1,40}?)(?:\s+(?:to|about|that)\b|[,.;]|$)", q, re.IGNORECASE)
        m = re.search(r"\b(?:to|about|that)\s+(.+)$", q, re.IGNORECASE)
        calls.append(
            {
                "name": "notify_team",
                "arguments": {
                    "recipient": _normalize(r.group(1)) if r else "Medical Team",
                    "message": _normalize(m.group(1)) if m else q,
                    "channel": "internal",
                },
            }
        )

    if ("dose" in ql or "dosage" in ql) and "calculate_medication_dose" in tool_names:
        w = re.search(r"\b(\d{1,3})\s*kg\b", q, re.IGNORECASE)
        a = re.search(r"\b(\d{1,2})\s*(?:year|yr)s?\b", q, re.IGNORECASE)
        calls.append(
            {
                "name": "calculate_medication_dose",
                "arguments": {
                    "patient_name": _extract_name(q),
                    "medication": _find_med(q),
                    "weight_kg": int(w.group(1)) if w else 0,
                    "age_years": int(a.group(1)) if a else 0,
                },
            }
        )

    return calls


def generate_hybrid(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    del confidence_threshold
    t0 = time.perf_counter()
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            query = _normalize(msg.get("content"))
            break
    if not query:
        return {
            "function_calls": [],
            "total_time_ms": 1.0,
            "source": "cactus-general",
            "router_profile": {"mode": "empty_query"},
        }

    fast = _sanitize_calls(_fastpath_calls(query, _tool_names(tools)), tools, query)
    if fast:
        return {
            "function_calls": fast,
            "total_time_ms": max(1.0, (time.perf_counter() - t0) * 1000.0),
            "source": "cactus-general-fastpath",
            "router_profile": {"mode": "fastpath", "clauses": len(_clauses(query))},
        }

    index = _build_index(tools)
    clauses = _clauses(query)
    chosen: Dict[str, Dict[str, Any]] = {}
    for clause in clauses:
        for tool in _top_tools_for_clause(clause, index, k=2):
            chosen[str(tool.get("name", ""))] = tool
    selected_tools = list(chosen.values())[:8] if chosen else tools[:8]

    try:
        first = _call_cactus(query, selected_tools)
        calls = _sanitize_calls(first.get("function_calls", []), tools, query)
        if not calls:
            calls = _semantic_fallback(query, tools)
            source = "semantic-fallback"
        else:
            source = "cactus-general"
        return {
            "function_calls": calls,
            "total_time_ms": max(1.0, (time.perf_counter() - t0) * 1000.0),
            "source": source,
            "router_profile": {
                "mode": "generalized",
                "selected_tool_count": len(selected_tools),
                "clauses": len(clauses),
            },
        }
    except Exception as exc:
        calls = _semantic_fallback(query, tools)
        return {
            "function_calls": calls,
            "total_time_ms": max(1.0, (time.perf_counter() - t0) * 1000.0),
            "source": "semantic-fallback",
            "router_profile": {
                "mode": "cactus_error_fallback",
                "error": str(exc),
            },
        }
