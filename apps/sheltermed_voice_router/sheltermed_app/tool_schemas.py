from typing import Any, Dict, List


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "register_patient",
        "description": "Register a patient in the field shelter intake queue",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string", "description": "Patient full name"},
                "age_years": {"type": "integer", "description": "Patient age in years"},
                "main_complaint": {"type": "string", "description": "Primary complaint or symptom"},
                "location_zone": {"type": "string", "description": "Shelter zone or triage area"},
            },
            "required": ["patient_name", "main_complaint"],
        },
    },
    {
        "name": "record_vitals",
        "description": "Record patient vitals for clinical monitoring",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string", "description": "Patient name"},
                "heart_rate_bpm": {"type": "integer", "description": "Heart rate beats per minute"},
                "spo2_percent": {"type": "integer", "description": "Oxygen saturation percentage"},
                "temperature_c": {"type": "string", "description": "Body temperature in Celsius"},
            },
            "required": ["patient_name"],
        },
    },
    {
        "name": "assign_triage_level",
        "description": "Assign triage level to patient (green/yellow/orange/red)",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string", "description": "Patient name"},
                "triage_level": {"type": "string", "description": "Triage category"},
                "reason": {"type": "string", "description": "Clinical reason for triage level"},
            },
            "required": ["patient_name", "triage_level"],
        },
    },
    {
        "name": "lookup_treatment_protocol",
        "description": "Retrieve emergency treatment protocol for a condition",
        "parameters": {
            "type": "object",
            "properties": {
                "condition": {"type": "string", "description": "Condition or syndrome"},
                "severity": {"type": "string", "description": "Severity if known"},
            },
            "required": ["condition"],
        },
    },
    {
        "name": "calculate_medication_dose",
        "description": "Calculate medication dose for a patient by weight and age",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string", "description": "Patient name"},
                "medication": {"type": "string", "description": "Medication name"},
                "weight_kg": {"type": "string", "description": "Weight in kilograms"},
                "age_years": {"type": "integer", "description": "Age in years"},
            },
            "required": ["medication"],
        },
    },
    {
        "name": "check_inventory",
        "description": "Check available stock of medication or medical supplies",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "Supply or medication item name"},
                "location_zone": {"type": "string", "description": "Storage or ward zone"},
            },
            "required": ["item"],
        },
    },
    {
        "name": "request_restock",
        "description": "Create a restock request for low inventory item",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "Item to restock"},
                "quantity": {"type": "integer", "description": "Requested quantity"},
                "priority": {"type": "string", "description": "low, medium, high, urgent"},
            },
            "required": ["item", "quantity"],
        },
    },
    {
        "name": "assign_bed",
        "description": "Assign a patient to a shelter bed or observation area",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string", "description": "Patient name"},
                "bed_zone": {"type": "string", "description": "Bed location or observation zone"},
            },
            "required": ["patient_name", "bed_zone"],
        },
    },
    {
        "name": "create_referral",
        "description": "Create referral to higher-level care facility",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string", "description": "Patient name"},
                "destination_facility": {"type": "string", "description": "Referral destination"},
                "reason": {"type": "string", "description": "Referral reason"},
                "urgency": {"type": "string", "description": "Routine, urgent, critical"},
            },
            "required": ["patient_name", "destination_facility", "reason"],
        },
    },
    {
        "name": "dispatch_transport",
        "description": "Dispatch transport for patient transfer or emergency evacuation",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string", "description": "Patient name"},
                "destination": {"type": "string", "description": "Transport destination"},
                "transport_type": {"type": "string", "description": "ambulance, van, rapid response"},
            },
            "required": ["patient_name", "destination"],
        },
    },
    {
        "name": "notify_team",
        "description": "Send a message to a clinician or team",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Person or team name"},
                "message": {"type": "string", "description": "Message content"},
                "channel": {"type": "string", "description": "radio, sms, internal"},
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "set_followup_timer",
        "description": "Set reassessment timer for patient review",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string", "description": "Patient name"},
                "minutes": {"type": "integer", "description": "Minutes until reassessment"},
                "task": {"type": "string", "description": "Task to perform at follow-up"},
            },
            "required": ["minutes"],
        },
    },
    {
        "name": "log_incident",
        "description": "Log major incident for shelter operations and medical oversight",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Incident title"},
                "severity": {"type": "string", "description": "Info, warning, critical"},
                "details": {"type": "string", "description": "Incident details"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "broadcast_alert",
        "description": "Broadcast alert to all shelter teams",
        "parameters": {
            "type": "object",
            "properties": {
                "alert_level": {"type": "string", "description": "advisory, urgent, critical"},
                "message": {"type": "string", "description": "Alert message"},
            },
            "required": ["alert_level", "message"],
        },
    },
    {
        "name": "create_shift_handoff",
        "description": "Create shift handoff summary for incoming medical staff",
        "parameters": {
            "type": "object",
            "properties": {
                "shift_name": {"type": "string", "description": "Shift label"},
                "highlight": {"type": "string", "description": "Key handoff highlights"},
            },
            "required": ["shift_name"],
        },
    },
]

