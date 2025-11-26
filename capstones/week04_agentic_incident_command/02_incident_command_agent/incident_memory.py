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
        self._data = initial_data or {}

    def list_resources(self) -> List[Dict[str, Any]]:
        """Return descriptors for available memory:// URIs."""
        raise NotImplementedError

    def get_resource(self, uri: str, cursor: Optional[str] = None) -> Dict[str, Any]:
        """Fetch resource payload by URI with optional cursor support."""
        raise NotImplementedError

    def write_delta(self, delta: Dict[str, Any]) -> Dict[str, Any]:
        """Append an action delta to memory://deltas/recent."""
        raise NotImplementedError

    def write_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Append evidence to memory://evidence/{id}."""
        raise NotImplementedError

    def update_incident(self, incident_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Update incident state in memory://incidents/{id}."""
        raise NotImplementedError

    def write_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store the current plan in memory://plans/current."""
        raise NotImplementedError

