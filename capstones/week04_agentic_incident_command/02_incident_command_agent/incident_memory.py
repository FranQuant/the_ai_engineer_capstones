"""
Memory layer for incident command agent using memory:// URIs.

Responsibilities:
- Serve typed resources (alerts, incidents, evidence, deltas, plans, transcripts).
- Provide read access with optional cursors/versions.
- Support append/update operations for deltas, evidence, incidents, and plans.
- Surface resource registry for MCP server initialization.

TODO:
- Implement persistence/backing store strategy.
- Enforce versioning and conflict detection on writes.
- Add cursor handling for incremental resource reads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class IncidentMemoryStore:
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None) -> None:
        """Instantiate the memory store with optional seed data."""
        self._data = {
            "incidents": {},
            "evidence": {},
            "deltas": [],
            "plans": {},
        }
        if initial_data:
            self._data["incidents"].update(initial_data.get("incidents", {}))
            self._data["evidence"].update(initial_data.get("evidence", {}))
            self._data["deltas"].extend(initial_data.get("deltas", []))
            self._data["plans"].update(initial_data.get("plans", {}))

    # Compatibility helpers -------------------------------------------------
    def read(self, uri: str, cursor: Optional[str] = None) -> Dict[str, Any]:
        """Alias for get_resource for callers expecting a read method."""
        return self.get_resource(uri, cursor)

    def write(self, uri: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Basic write router used by callers that pass explicit URIs."""
        if uri.startswith("memory://incidents/"):
            incident_id = uri.rsplit("/", 1)[-1]
            return self.update_incident(incident_id, payload)
        if uri.startswith("memory://evidence/"):
            return self.write_evidence(payload)
        if uri == "memory://deltas/recent":
            return self.write_delta(payload)
        if uri == "memory://plans/current":
            return self.write_plan(payload)  # type: ignore[arg-type]
        raise ValueError(f"Unsupported write URI: {uri}")

    def list_resources(self) -> List[Dict[str, Any]]:
        """Return descriptors for available memory:// URIs."""
        return [
            {"uri": "memory://incidents/{id}", "type": "incident"},
            {"uri": "memory://evidence/{id}", "type": "evidence"},
            {"uri": "memory://deltas/recent", "type": "delta_list"},
            {"uri": "memory://plans/current", "type": "plan"},
        ]

    def get_resource(self, uri: str, cursor: Optional[str] = None) -> Dict[str, Any]:
        """Fetch resource payload by URI with optional cursor support."""
        del cursor  # no cursor semantics in minimal implementation
        if uri.startswith("memory://incidents/"):
            incident_id = uri.rsplit("/", 1)[-1]
            return self._data["incidents"].get(incident_id, {})
        if uri.startswith("memory://evidence/"):
            evidence_id = uri.rsplit("/", 1)[-1]
            return self._data["evidence"].get(evidence_id, {})
        if uri == "memory://deltas/recent":
            return {"items": list(self._data["deltas"])}
        if uri == "memory://plans/current":
            return {"plan": self._data["plans"].get("current", [])}
        raise ValueError(f"Unknown resource URI: {uri}")

    def write_delta(self, delta: Dict[str, Any]) -> Dict[str, Any]:
        """Append an action delta to memory://deltas/recent."""
        self._data["deltas"].append(delta)
        return delta

    def append_delta(self, delta: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for write_delta for compatibility."""
        return self.write_delta(delta)

    def write_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Append evidence to memory://evidence/{id}."""
        evidence_id = evidence.get("id") or str(len(self._data["evidence"]) + 1)
        evidence_with_id = {**evidence, "id": evidence_id}
        self._data["evidence"][evidence_id] = evidence_with_id
        return evidence_with_id

    def update_incident(self, incident_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Update incident state in memory://incidents/{id}."""
        incident = self._data["incidents"].get(incident_id, {"id": incident_id})
        incident.update(fields)
        self._data["incidents"][incident_id] = incident
        return incident

    def write_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store the current plan in memory://plans/current."""
        self._data["plans"]["current"] = list(plan)
        return {"plan": list(plan)}
