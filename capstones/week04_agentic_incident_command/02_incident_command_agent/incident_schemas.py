"""
JSON schemas for tools and resources used by the Incident Command Agent.

Responsibilities:
- Define input/output schemas for all MCP tools.
- Define resource shapes for memory:// URIs.
- Provide registry helpers for server capabilities payload.

TODO:
- Populate detailed schemas for metrics, errors, and citations.
- Align resource schemas with memory store serialization format.
"""

from __future__ import annotations

from typing import Dict, List


def get_tool_schemas() -> Dict[str, Dict[str, object]]:
    """Return mapping of tool name to JSON schema definitions."""
    return {
        "create_incident": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "severity": {"type": "string"},
            },
            "required": ["id", "title"],
        },
        "add_evidence": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": {"type": "string"},
                "source": {"type": "string"},
            },
            "required": ["content"],
        },
        "append_delta": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "details": {"type": "object"},
            },
            "required": ["action"],
        },
    }


def get_resource_schemas() -> Dict[str, Dict[str, object]]:
    """Return mapping of memory:// resource URIs to JSON schema definitions."""
    return {
        "memory://incidents/{id}": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "severity": {"type": "string"},
            },
            "required": ["id"],
        },
        "memory://evidence/{id}": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": {"type": "string"},
                "source": {"type": "string"},
            },
            "required": ["id", "content"],
        },
        "memory://deltas/recent": {
            "type": "object",
            "properties": {
                "items": {"type": "array"},
            },
            "required": ["items"],
        },
        "memory://plans/current": {
            "type": "object",
            "properties": {
                "plan": {"type": "array"},
            },
            "required": ["plan"],
        },
    }


def tool_descriptions() -> List[Dict[str, object]]:
    """Return tool descriptors combining name, description, and schema."""
    schemas = get_tool_schemas()
    return [
        {
            "name": name,
            "description": f"Stub tool for {name.replace('_', ' ')}",
            "inputSchema": schema,
        }
        for name, schema in schemas.items()
    ]


def resource_descriptions() -> List[Dict[str, object]]:
    """Return resource descriptors for server capabilities."""
    schemas = get_resource_schemas()
    return [
        {"uri": uri, "schema": schema, "description": f"Resource for {uri}"}
        for uri, schema in schemas.items()
    ]
