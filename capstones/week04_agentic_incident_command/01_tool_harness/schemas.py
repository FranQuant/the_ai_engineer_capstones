"""
JSON schemas and resource stubs for the MCP tool harness warm-up.
Aligned with TAE Week-04, MCP Agents, and engcode-main patterns.
"""

from __future__ import annotations
from typing import Dict, List

# ============================================================
# Tool JSON Schemas
# ============================================================

RETRIEVE_RUNBOOK_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 2},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
    },
    "required": ["query"],
}

RUN_DIAGNOSTIC_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "host": {"type": "string"},
    },
    "required": ["command", "host"],
}

SUMMARIZE_INCIDENT_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "alert_id": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
    },
    "required": ["alert_id"],
}

TOOL_SCHEMAS: Dict[str, Dict[str, object]] = {
    "retrieve_runbook": RETRIEVE_RUNBOOK_SCHEMA,
    "run_diagnostic": RUN_DIAGNOSTIC_SCHEMA,
    "summarize_incident": SUMMARIZE_INCIDENT_SCHEMA,
}

# ============================================================
# Resource fixtures (typed examples for the warm-up server)
# ============================================================

RESOURCE_FIXTURES: Dict[str, object] = {
    "memory://alerts/latest": {
        "alert": {
            "id": "ALRT-0001",
            "service": "staging-api",
            "symptom": "CPU spike on node-3",
            "severity": "high",
            "detected_at": "2025-11-23T09:00:00Z",
        },
        "recommendations": [
            "Restart service if CPU > 90% for 5 minutes.",
            "Capture pod logs before restart.",
        ],
    },

    "memory://runbooks/index": [
        {"id": "rb-101", "title": "High CPU playbook", "service": "staging-api"},
        {"id": "rb-202", "title": "Crashloop restart", "service": "staging-api"},
    ],

    "memory://deltas/recent": [
        {
            "alert_id": "ALRT-0001",
            "action": "observe",
            "summary": "Initial triage; severity high.",
            "timestamp": "2025-11-23T09:01:00Z",
        }
    ],
}

# Convenience: list of all resource URIs
RESOURCE_URIS: List[str] = list(RESOURCE_FIXTURES.keys())
