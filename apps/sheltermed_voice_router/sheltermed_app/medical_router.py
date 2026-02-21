from __future__ import annotations

import importlib.util
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / ".git").exists():
            return parent
    return start.resolve()


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())
_STATE: Dict[str, Any] = {
    "base_router_path": None,
    "last_mtime": 0.0,
    "loaded_at": 0.0,
    "generate_hybrid": None,
}
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
}
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


def _router_candidates(repo_root: Path) -> List[Path]:
    return [
        repo_root / "Resources" / "functiongemma-hackathon" / "main.py",
        repo_root / "Resources" / "functiongemma-hackathon-hybrid-reset" / "main.py",
        repo_root / "Resources" / "functiongemma-hackathon-hybrid-reset-stale" / "main.py",
    ]


def _resolve_base_router_path() -> Path:
    override = os.environ.get("SHELTERMED_BASE_ROUTER_PATH", "").strip()
    if override:
        path = Path(override)
        return path if path.is_absolute() else (_REPO_ROOT / path).resolve()

    candidates = [p for p in _router_candidates(_REPO_ROOT) if p.exists()]
    if not candidates:
        raise RuntimeError("No base router found under Resources/")

    # Prefer whichever router file was updated most recently.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_base_router() -> None:
    path = _resolve_base_router_path()
    mtime = path.stat().st_mtime
    if (
        _STATE["generate_hybrid"] is not None
        and _STATE["base_router_path"] == path
        and mtime <= _STATE["last_mtime"]
    ):
        return

    module_name = f"sheltermed_base_router_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load router spec: {path}")

    module = importlib.util.module_from_spec(spec)
    old_cwd = Path.cwd()
    try:
        os.chdir(_REPO_ROOT)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)

    fg_path = getattr(module, "functiongemma_path", None)
    if isinstance(fg_path, str) and fg_path and not Path(fg_path).is_absolute():
        module.functiongemma_path = str((_REPO_ROOT / fg_path).resolve())

    route_fn = getattr(module, "generate_hybrid", None)
    if not callable(route_fn):
        raise RuntimeError(f"Base router has no callable generate_hybrid: {path}")

    _STATE["base_router_path"] = path
    _STATE["last_mtime"] = mtime
    _STATE["loaded_at"] = time.time()
    _STATE["generate_hybrid"] = route_fn


def _last_user_query(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return " ".join(str(msg.get("content", "")).split())
    return ""


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value or "").strip().lower()
    if not text:
        return default
    m = re.search(r"-?\d+", text)
    if m:
        try:
            return int(m.group(0))
        except ValueError:
            return default
    if text in _NUMBER_WORDS:
        return _NUMBER_WORDS[text]
    return default


def _titlecase_name(text: str) -> str:
    s = " ".join(str(text or "").split())
    if not s:
        return s
    return " ".join(token[:1].upper() + token[1:] for token in s.split(" "))


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


def _find_priority(query: str) -> str:
    q = query.lower()
    if "critical" in q:
        return "urgent"
    if "urgent" in q:
        return "urgent"
    if "high" in q:
        return "high"
    if "low" in q:
        return "low"
    return "medium"


def _find_medication(query: str) -> str:
    q = query.lower()
    for med in _KNOWN_MEDS:
        if med in q:
            return med
    return "paracetamol"


def _extract_first_number(query: str, default: int = 0) -> int:
    m = re.search(r"\b(\d{1,4})\b", query)
    if m:
        return _safe_int(m.group(1), default=default)
    for token in re.findall(r"[A-Za-z]+", query):
        if token.lower() in _NUMBER_WORDS:
            return _NUMBER_WORDS[token.lower()]
    return default


def _extract_patient_name(query: str) -> str:
    patterns = [
        r"\bfor\s+([A-Z][A-Za-z' -]{1,40}?)(?:\s+(?:with|in|at|to|and)\b|[,.;]|$)",
        r"\bpatient\s+([A-Z][A-Za-z' -]{1,40}?)(?:\s+(?:with|for|in|at|and)\b|[,.;]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, query)
        if m:
            return _titlecase_name(m.group(1).strip(" ,.;"))
    return "Unknown Patient"


def _fastpath_calls(query: str, tool_names: set[str]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    q = query
    ql = query.lower()
    registered_name: Optional[str] = None

    if "register" in ql and "patient" in ql and "register_patient" in tool_names:
        name_match = re.search(
            r"\bregister(?:\s+patient)?\s+([A-Za-z][A-Za-z' -]{1,40}?)(?:\s+(?:with|for|in|at)\b|[,.;]|$)",
            q,
            re.IGNORECASE,
        )
        complaint_match = re.search(
            r"\bwith\s+(.+?)(?:\s+(?:in\s+zone|assign|set|notify|and|then)\b|[,.;]|$)",
            q,
            re.IGNORECASE,
        )
        zone_match = re.search(r"\bzone\s+([A-Za-z0-9-]+)\b", q, re.IGNORECASE)
        calls.append(
            {
                "name": "register_patient",
                "arguments": {
                    "patient_name": _titlecase_name(name_match.group(1)) if name_match else "Unknown Patient",
                    "main_complaint": complaint_match.group(1).strip() if complaint_match else "unspecified complaint",
                    "location_zone": f"Zone {zone_match.group(1).upper()}" if zone_match else "intake",
                },
            }
        )
        registered_name = _titlecase_name(name_match.group(1)) if name_match else None

    if (
        ("inventory" in ql or "stock" in ql or "supply" in ql)
        and "check_inventory" in tool_names
    ):
        calls.append({"name": "check_inventory", "arguments": {"item": _find_item(q)}})

    if ("restock" in ql or "resupply" in ql) and "request_restock" in tool_names:
        calls.append(
            {
                "name": "request_restock",
                "arguments": {
                    "item": _find_item(q),
                    "quantity": max(1, _extract_first_number(q, default=10)),
                    "priority": _find_priority(q),
                },
            }
        )

    triage_match = re.search(r"\b(red|orange|yellow|green)\s+triage\b", q, re.IGNORECASE)
    if triage_match and "assign_triage_level" in tool_names:
        calls.append(
            {
                "name": "assign_triage_level",
                "arguments": {
                    "patient_name": registered_name or _extract_patient_name(q),
                    "triage_level": triage_match.group(1).lower(),
                    "reason": "Set from spoken triage command",
                },
            }
        )

    if ("notify" in ql or "message" in ql) and "notify_team" in tool_names:
        recipient_match = re.search(
            r"\bnotify\s+(?:the\s+)?([A-Za-z][A-Za-z .'-]{1,40}?)(?:\s+(?:to|about|that)\b|[,.;]|$)",
            q,
            re.IGNORECASE,
        )
        msg_match = re.search(r"\b(?:to|about|that)\s+(.+)$", q, re.IGNORECASE)
        calls.append(
            {
                "name": "notify_team",
                "arguments": {
                    "recipient": _titlecase_name(recipient_match.group(1)) if recipient_match else "Medical Team",
                    "message": msg_match.group(1).strip() if msg_match else q,
                    "channel": "internal",
                },
            }
        )

    if ("follow-up" in ql or "follow up" in ql or "reassess" in ql) and "set_followup_timer" in tool_names:
        calls.append(
            {
                "name": "set_followup_timer",
                "arguments": {
                    "patient_name": registered_name or _extract_patient_name(q),
                    "minutes": max(1, _extract_first_number(q, default=20)),
                    "task": "reassess",
                },
            }
        )

    if ("referral" in ql or "refer" in ql) and "create_referral" in tool_names:
        dest_match = re.search(r"\bto\s+([A-Za-z][A-Za-z0-9 .'-]{2,60})(?:,|\.|$)", q, re.IGNORECASE)
        calls.append(
            {
                "name": "create_referral",
                "arguments": {
                    "patient_name": _extract_patient_name(q),
                    "destination_facility": _titlecase_name(dest_match.group(1).strip()) if dest_match else "Regional Hospital",
                    "reason": "Needs higher-level evaluation",
                    "urgency": _find_priority(q),
                },
            }
        )

    if ("transport" in ql or "ambulance" in ql or "evacuate" in ql) and "dispatch_transport" in tool_names:
        transport = "ambulance" if "ambulance" in ql else "van"
        dest_match = re.search(r"\bto\s+([A-Za-z][A-Za-z0-9 .'-]{2,60})(?:,|\.|$)", q, re.IGNORECASE)
        calls.append(
            {
                "name": "dispatch_transport",
                "arguments": {
                    "patient_name": _extract_patient_name(q),
                    "destination": _titlecase_name(dest_match.group(1).strip()) if dest_match else "Regional Hospital",
                    "transport_type": transport,
                },
            }
        )

    if (
        (("record" in ql and "vital" in ql) or "heart rate" in ql or "spo2" in ql or "temperature" in ql)
        and "record_vitals" in tool_names
    ):
        hr_match = re.search(r"\b(?:heart\s*rate|hr)\s*(\d{2,3})\b", q, re.IGNORECASE)
        spo2_match = re.search(r"\b(?:spo2|oxygen)\s*(\d{2,3})\b", q, re.IGNORECASE)
        temp_match = re.search(r"\btemp(?:erature)?\s*([0-9]+(?:\.[0-9]+)?)\b", q, re.IGNORECASE)
        calls.append(
            {
                "name": "record_vitals",
                "arguments": {
                    "patient_name": _extract_patient_name(q),
                    "heart_rate_bpm": _safe_int(hr_match.group(1), default=0) if hr_match else 0,
                    "spo2_percent": _safe_int(spo2_match.group(1), default=0) if spo2_match else 0,
                    "temperature_c": temp_match.group(1) if temp_match else "",
                },
            }
        )

    if ("dose" in ql or "dosage" in ql or "medication" in ql) and "calculate_medication_dose" in tool_names:
        weight_match = re.search(r"\b(\d{1,3})\s*kg\b", q, re.IGNORECASE)
        age_match = re.search(r"\b(\d{1,2})\s*(?:year|yr)s?\b", q, re.IGNORECASE)
        calls.append(
            {
                "name": "calculate_medication_dose",
                "arguments": {
                    "patient_name": _extract_patient_name(q),
                    "medication": _find_medication(q),
                    "weight_kg": _safe_int(weight_match.group(1), default=0) if weight_match else 0,
                    "age_years": _safe_int(age_match.group(1), default=0) if age_match else 0,
                },
            }
        )

    return calls


def _coerce_args(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            import json

            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _sanitize_calls(
    function_calls: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    tool_names = {str(t.get("name", "")) for t in tools}
    required_by_tool: Dict[str, List[str]] = {
        str(t.get("name", "")): list(t.get("parameters", {}).get("required", []))
        for t in tools
        if isinstance(t, dict)
    }
    out: List[Dict[str, Any]] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    q = query
    ql = query.lower()

    for raw in function_calls:
        if not isinstance(raw, dict):
            continue
        name = " ".join(str(raw.get("name", "")).split())
        if not name or name not in tool_names:
            continue

        args = _coerce_args(raw.get("arguments", {}))
        if name == "check_inventory":
            item = " ".join(str(args.get("item", "")).split()).lower()
            if item in {"", "check", "inventory", "stock", "supply", "supplies"}:
                args["item"] = _find_item(q)

        elif name == "request_restock":
            item = " ".join(str(args.get("item", "")).split()).lower()
            if item in {"", "restock", "urgent restock", "stock", "supply"}:
                args["item"] = _find_item(q)
            qty = _safe_int(args.get("quantity"), default=0)
            args["quantity"] = qty if qty > 0 else max(1, _extract_first_number(q, default=10))
            args["priority"] = _find_priority(str(args.get("priority", "")) + " " + q)

        elif name == "notify_team":
            recipient = " ".join(str(args.get("recipient", "")).split()).strip()
            if recipient.lower() in {"", "check", "team", "medical team"}:
                m = re.search(r"\bnotify\s+(?:the\s+)?([A-Za-z][A-Za-z .'-]{1,40}?)(?:\s+(?:to|about|that)\b|[,.;]|$)", q, re.IGNORECASE)
                recipient = _titlecase_name(m.group(1)) if m else "Medical Team"
            message = " ".join(str(args.get("message", "")).split()).strip()
            if len(message) < 5:
                m2 = re.search(r"\b(?:to|about|that)\s+(.+)$", q, re.IGNORECASE)
                message = m2.group(1).strip() if m2 else q
            channel = " ".join(str(args.get("channel", "")).split()).lower()
            if channel not in {"internal", "radio", "sms"}:
                channel = "internal"
            args = {"recipient": recipient, "message": message, "channel": channel}

        elif name == "register_patient":
            patient_name = " ".join(str(args.get("patient_name", "")).split()).strip()
            patient_name = re.sub(r"^(patient\s+)", "", patient_name, flags=re.IGNORECASE).strip()
            if len(patient_name) < 2:
                patient_name = _extract_patient_name(q)
            complaint = " ".join(str(args.get("main_complaint", "")).split()).strip()
            if len(complaint) < 3 or complaint.lower() in {"zone", "triage"}:
                m = re.search(r"\bwith\s+(.+?)(?:\s+(?:in\s+zone|assign|set|notify|and|then)\b|[,.;]|$)", q, re.IGNORECASE)
                complaint = m.group(1).strip() if m else "unspecified complaint"
            zone = " ".join(str(args.get("location_zone", "")).split()).strip()
            if not zone:
                mz = re.search(r"\bzone\s+([A-Za-z0-9-]+)\b", q, re.IGNORECASE)
                zone = f"Zone {mz.group(1).upper()}" if mz else "intake"
            args = {"patient_name": _titlecase_name(patient_name), "main_complaint": complaint, "location_zone": zone}

        elif name == "assign_triage_level":
            triage = " ".join(str(args.get("triage_level", "")).split()).lower()
            for level in ("red", "orange", "yellow", "green"):
                if level in triage or level in ql:
                    triage = level
                    break
            if triage not in {"red", "orange", "yellow", "green"}:
                triage = "yellow"
            patient_name = " ".join(str(args.get("patient_name", "")).split()).strip()
            if not patient_name or patient_name.lower() in {"triage", "red triage", "yellow triage"}:
                patient_name = _extract_patient_name(q)
            reason = " ".join(str(args.get("reason", "")).split()).strip() or "Set from command"
            args = {"patient_name": _titlecase_name(patient_name), "triage_level": triage, "reason": reason}

        elif name == "set_followup_timer":
            mins = _safe_int(args.get("minutes"), default=0)
            args["minutes"] = mins if mins > 0 else max(1, _extract_first_number(q, default=20))
            patient_name = " ".join(str(args.get("patient_name", "")).split()).strip()
            if len(patient_name) < 2:
                args["patient_name"] = _extract_patient_name(q)
            task = " ".join(str(args.get("task", "")).split()).strip()
            if len(task) < 2:
                args["task"] = "reassess"

        elif name == "calculate_medication_dose":
            med = " ".join(str(args.get("medication", "")).split()).lower()
            if med not in _KNOWN_MEDS:
                med = _find_medication(q)
            weight = _safe_int(args.get("weight_kg"), default=0)
            if weight <= 0:
                m_w = re.search(r"\b(\d{1,3})\s*kg\b", q, re.IGNORECASE)
                weight = _safe_int(m_w.group(1), default=0) if m_w else 0
            age = _safe_int(args.get("age_years"), default=0)
            if age <= 0:
                m_a = re.search(r"\b(\d{1,2})\s*(?:year|yr)s?\b", q, re.IGNORECASE)
                age = _safe_int(m_a.group(1), default=0) if m_a else 0
            patient_name = " ".join(str(args.get("patient_name", "")).split()).strip()
            if len(patient_name) < 2:
                patient_name = _extract_patient_name(q)
            args = {
                "patient_name": _titlecase_name(patient_name),
                "medication": med,
                "weight_kg": weight,
                "age_years": age,
            }

        required = required_by_tool.get(name, [])
        for req in required:
            if req not in args or args[req] in {"", None}:
                if req == "item":
                    args[req] = _find_item(q)
                elif req == "quantity":
                    args[req] = max(1, _extract_first_number(q, default=10))
                elif req == "message":
                    args[req] = q
                elif req == "recipient":
                    args[req] = "Medical Team"
                elif req == "patient_name":
                    args[req] = _extract_patient_name(q)
                elif req == "minutes":
                    args[req] = max(1, _extract_first_number(q, default=20))
                else:
                    args[req] = "unspecified"

        sig = (name, tuple(sorted((k, str(v).lower()) for k, v in args.items())))
        if sig in seen:
            continue
        seen.add(sig)
        out.append({"name": name, "arguments": args})

    return out


def generate_hybrid(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    query = _last_user_query(messages)
    tool_names = {str(t.get("name", "")) for t in tools if isinstance(t, dict)}

    fast_start = time.perf_counter()
    fast_calls = _fastpath_calls(query, tool_names)
    fast_calls = _sanitize_calls(fast_calls, tools, query) if fast_calls else []
    if fast_calls:
        return {
            "function_calls": fast_calls,
            "total_time_ms": max(1.0, (time.perf_counter() - fast_start) * 1000.0),
            "source": "medical-fastpath",
            "router_profile": {
                "wrapper": "sheltermed_medical_router",
                "base_router_path": str(_resolve_base_router_path()),
                "loaded_at": _STATE["loaded_at"],
                "mode": "fastpath",
            },
        }

    _load_base_router()
    route_fn: Optional[Callable[..., Dict[str, Any]]] = _STATE["generate_hybrid"]
    if route_fn is None:
        raise RuntimeError("Base router is not loaded")

    start = time.perf_counter()
    try:
        result = route_fn(messages, tools, confidence_threshold=confidence_threshold)
    except TypeError:
        result = route_fn(messages, tools)
    except Exception:
        result = {}

    if not isinstance(result, dict):
        result = {}

    calls = result.get("function_calls", [])
    if not isinstance(calls, list):
        calls = []
    calls = _sanitize_calls(calls, tools, query)

    total_time_ms = result.get("total_time_ms", 0.0)
    if not isinstance(total_time_ms, (int, float)) or total_time_ms <= 0:
        total_time_ms = (time.perf_counter() - start) * 1000.0

    return {
        **result,
        "function_calls": calls,
        "total_time_ms": max(1.0, float(total_time_ms)),
        "source": result.get("source", "router"),
        "router_profile": {
            "wrapper": "sheltermed_medical_router",
            "base_router_path": str(_STATE["base_router_path"]),
            "loaded_at": _STATE["loaded_at"],
            "mode": "delegated",
        },
    }
