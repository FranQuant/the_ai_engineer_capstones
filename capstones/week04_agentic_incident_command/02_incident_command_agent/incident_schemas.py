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
    raise NotImplementedError


def get_resource_schemas() -> Dict[str, Dict[str, object]]:
    """Return mapping of memory:// resource URIs to JSON schema definitions."""
    raise NotImplementedError


def tool_descriptions() -> List[Dict[str, object]]:
    """Return tool descriptors combining name, description, and schema."""
    raise NotImplementedError


def resource_descriptions() -> List[Dict[str, object]]:
    """Return resource descriptors for server capabilities."""
    raise NotImplementedError

