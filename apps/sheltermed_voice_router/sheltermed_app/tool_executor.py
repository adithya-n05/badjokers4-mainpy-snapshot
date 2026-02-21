from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


def _to_int(v: Any, default: int = 0) -> int:
    try:
        if isinstance(v, bool):
            return default
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v)
        digits = "".join(ch for ch in s if ch.isdigit() or ch == "-")
        return int(digits) if digits else default
    except Exception:
        return default


def _norm(s: Any) -> str:
    return " ".join(str(s or "").strip().split())


@dataclass
class ShelterState:
    patients: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    triage_queue: List[Dict[str, Any]] = field(default_factory=list)
    referrals: List[Dict[str, Any]] = field(default_factory=list)
    transport_jobs: List[Dict[str, Any]] = field(default_factory=list)
    followups: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    incidents: List[Dict[str, Any]] = field(default_factory=list)
    shifts: List[Dict[str, Any]] = field(default_factory=list)
    inventory: Dict[str, int] = field(
        default_factory=lambda: {
            "paracetamol": 120,
            "ibuprofen": 80,
            "salbutamol": 34,
            "oxygen cylinder": 12,
            "iv fluids": 40,
            "bandages": 230,
            "gloves": 900,
            "antibiotics": 54,
        }
    )
    activity_log: List[Dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "patient_count": len(self.patients),
            "triage_queue_count": len(self.triage_queue),
            "referral_count": len(self.referrals),
            "transport_jobs_count": len(self.transport_jobs),
            "followup_count": len(self.followups),
            "message_count": len(self.messages),
            "incident_count": len(self.incidents),
            "shift_count": len(self.shifts),
            "inventory": self.inventory,
            "latest_activities": self.activity_log[-12:],
        }


class ActionExecutor:
    def __init__(self) -> None:
        self.state = ShelterState()

    def execute_many(self, function_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outcomes: List[Dict[str, Any]] = []
        for call in function_calls:
            outcomes.append(self.execute_one(call))
        return outcomes

    def execute_one(self, call: Dict[str, Any]) -> Dict[str, Any]:
        name = _norm(call.get("name"))
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}

        handler = getattr(self, f"_run_{name}", None)
        if handler is None:
            result = {
                "name": name,
                "success": False,
                "message": f"Unsupported action `{name}`.",
                "arguments": args,
                "data": {},
            }
            self._log(result)
            return result

        try:
            message, data = handler(args)
            result = {
                "name": name,
                "success": True,
                "message": message,
                "arguments": args,
                "data": data,
            }
            self._log(result)
            return result
        except Exception as exc:
            result = {
                "name": name,
                "success": False,
                "message": f"{name} failed: {exc}",
                "arguments": args,
                "data": {},
            }
            self._log(result)
            return result

    def _log(self, result: Dict[str, Any]) -> None:
        self.state.activity_log.append(
            {
                "timestamp": time.strftime("%H:%M:%S"),
                "name": result["name"],
                "success": result["success"],
                "message": result["message"],
            }
        )

    def _run_register_patient(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        name = _norm(args.get("patient_name")) or "Unknown Patient"
        complaint = _norm(args.get("main_complaint")) or "unspecified complaint"
        age = _to_int(args.get("age_years"), default=0)
        zone = _norm(args.get("location_zone")) or "intake"
        self.state.patients[name] = {
            "age_years": age if age > 0 else None,
            "complaint": complaint,
            "zone": zone,
        }
        self.state.triage_queue.append({"patient_name": name, "complaint": complaint, "zone": zone})
        return (
            f"Registered {name} in {zone} with complaint: {complaint}.",
            {"patient_name": name, "zone": zone},
        )

    def _run_record_vitals(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        name = _norm(args.get("patient_name")) or "Unknown Patient"
        hr = _to_int(args.get("heart_rate_bpm"), default=0)
        spo2 = _to_int(args.get("spo2_percent"), default=0)
        temp = _norm(args.get("temperature_c"))
        entry = {"heart_rate_bpm": hr or None, "spo2_percent": spo2 or None, "temperature_c": temp or None}
        self.state.patients.setdefault(name, {})["latest_vitals"] = entry
        return (f"Vitals recorded for {name}.", entry)

    def _run_assign_triage_level(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        name = _norm(args.get("patient_name")) or "Unknown Patient"
        level = (_norm(args.get("triage_level")) or "yellow").lower()
        reason = _norm(args.get("reason")) or "no reason specified"
        self.state.patients.setdefault(name, {})["triage_level"] = level
        self.state.patients[name]["triage_reason"] = reason
        return (f"Triage set to {level.upper()} for {name}.", {"patient_name": name, "triage_level": level})

    def _run_lookup_treatment_protocol(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        condition = (_norm(args.get("condition")) or "general emergency").lower()
        severity = (_norm(args.get("severity")) or "unknown").lower()
        protocols = {
            "dehydration": ["Oral rehydration", "Monitor pulse", "Escalate if persistent vomiting"],
            "asthma": ["Bronchodilator", "Oxygen support", "Reassess after 20 min"],
            "anaphylaxis": ["IM epinephrine", "Airway monitoring", "Urgent transfer"],
            "heat stroke": ["Rapid cooling", "Monitor mental status", "Urgent transfer if persistent symptoms"],
        }
        steps = protocols.get(condition, ["Stabilize airway/breathing/circulation", "Follow local clinical SOP"])
        return (
            f"Loaded protocol for {condition}.",
            {"condition": condition, "severity": severity, "steps": steps},
        )

    def _run_calculate_medication_dose(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        med = (_norm(args.get("medication")) or "paracetamol").lower()
        weight = float(_to_int(args.get("weight_kg"), default=0))
        age = _to_int(args.get("age_years"), default=0)
        patient = _norm(args.get("patient_name")) or "patient"

        mg_per_kg = {"paracetamol": 15, "ibuprofen": 10, "cetirizine": 0}
        default_dose = {"paracetamol": 500, "ibuprofen": 400, "cetirizine": 10}
        interval = {"paracetamol": 6, "ibuprofen": 8, "cetirizine": 24}

        if weight > 0 and med in mg_per_kg and mg_per_kg[med] > 0:
            dose = int(weight * mg_per_kg[med])
            method = "weight_based"
        else:
            dose = default_dose.get(med, 0)
            method = "default_adult" if age >= 12 else "estimate_needs_weight"

        data = {"patient_name": patient, "medication": med, "single_dose_mg": dose, "interval_hours": interval.get(med, 8), "method": method}
        return (f"Dose calculated: {dose} mg {med} for {patient}.", data)

    def _run_check_inventory(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        item = (_norm(args.get("item")) or "").lower()
        if not item:
            raise ValueError("missing item")
        qty = self.state.inventory.get(item, 0)
        status = "low" if qty < 20 else "ok"
        return (f"Inventory check: {item} has {qty} units ({status}).", {"item": item, "quantity": qty, "status": status})

    def _run_request_restock(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        item = (_norm(args.get("item")) or "").lower()
        qty = _to_int(args.get("quantity"), default=0)
        priority = (_norm(args.get("priority")) or "medium").lower()
        if not item or qty <= 0:
            raise ValueError("item and positive quantity required")
        msg = {"item": item, "quantity": qty, "priority": priority}
        self.state.messages.append({"type": "restock", **msg})
        return (f"Restock request created for {qty} {item} ({priority}).", msg)

    def _run_assign_bed(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        name = _norm(args.get("patient_name")) or "Unknown Patient"
        zone = _norm(args.get("bed_zone")) or "observation"
        self.state.patients.setdefault(name, {})["bed_zone"] = zone
        return (f"{name} assigned to {zone}.", {"patient_name": name, "bed_zone": zone})

    def _run_create_referral(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        entry = {
            "patient_name": _norm(args.get("patient_name")) or "Unknown Patient",
            "destination_facility": _norm(args.get("destination_facility")) or "Regional Hospital",
            "reason": _norm(args.get("reason")) or "Higher level evaluation required",
            "urgency": (_norm(args.get("urgency")) or "urgent").lower(),
        }
        self.state.referrals.append(entry)
        return (f"Referral prepared for {entry['patient_name']} to {entry['destination_facility']}.", entry)

    def _run_dispatch_transport(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        job = {
            "patient_name": _norm(args.get("patient_name")) or "Unknown Patient",
            "destination": _norm(args.get("destination")) or "Regional Hospital",
            "transport_type": (_norm(args.get("transport_type")) or "ambulance").lower(),
        }
        self.state.transport_jobs.append(job)
        return (f"Transport dispatched: {job['transport_type']} for {job['patient_name']}.", job)

    def _run_notify_team(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        msg = {
            "recipient": _norm(args.get("recipient")) or "Medical Team",
            "message": _norm(args.get("message")) or "Please review current case updates.",
            "channel": (_norm(args.get("channel")) or "internal").lower(),
        }
        self.state.messages.append(msg)
        return (f"Message sent to {msg['recipient']} via {msg['channel']}.", msg)

    def _run_set_followup_timer(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        patient = _norm(args.get("patient_name")) or "Unspecified Patient"
        minutes = max(1, _to_int(args.get("minutes"), default=15))
        task = _norm(args.get("task")) or "reassess"
        item = {"patient_name": patient, "minutes": minutes, "task": task}
        self.state.followups.append(item)
        return (f"Follow-up timer set: {minutes} min for {patient} ({task}).", item)

    def _run_log_incident(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        incident = {
            "title": _norm(args.get("title")) or "Unnamed Incident",
            "severity": (_norm(args.get("severity")) or "warning").lower(),
            "details": _norm(args.get("details")) or "",
        }
        self.state.incidents.append(incident)
        return (f"Incident logged: {incident['title']} ({incident['severity']}).", incident)

    def _run_broadcast_alert(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        level = (_norm(args.get("alert_level")) or "advisory").lower()
        message = _norm(args.get("message")) or "Please review command center updates."
        payload = {"alert_level": level, "message": message}
        self.state.messages.append({"recipient": "all_staff", "channel": "broadcast", **payload})
        return (f"Broadcast alert sent ({level}).", payload)

    def _run_create_shift_handoff(self, args: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        handoff = {
            "shift_name": _norm(args.get("shift_name")) or "Current Shift",
            "highlight": _norm(args.get("highlight")) or "No highlight provided.",
            "patients_active": len(self.state.patients),
            "referrals_pending": len(self.state.referrals),
        }
        self.state.shifts.append(handoff)
        return (f"Handoff summary created for {handoff['shift_name']}.", handoff)

