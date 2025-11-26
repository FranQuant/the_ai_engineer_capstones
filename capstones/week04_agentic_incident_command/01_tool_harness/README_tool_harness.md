# MCP Tool Harness (Warm-Up)

This directory implements the Tool Harness warm-up project for Week-04 of The AI Engineer program.
It provides a minimal, self-contained MCP server and client that exercise:

- JSON schema modeling
- Basic MCP tool capabilities
- Deterministic OPAL loops (Observe → Plan → Act → Learn)
- Telemetry logging
- Replay-friendly JSONL traces
- In-memory resource handling through `memory://` URIs

The warm-up is a simplified version of the architecture used in the full Incident Command Agent located in `02_incident_command_agent/`.

---

## Directory Structure

```
01_tool_harness/
│
├── schemas.py
├── telemetry.py
├── mcp_tool_harness_server.py
├── mcp_tool_harness_client.py
│
├── samples/
│   ├── client_telemetry.log
│   └── server_telemetry.log
│
└── README_tool_harness.md
```

---

## Component Overview

### schemas.py
Defines JSON schemas for the three warm-up tools:

- retrieve_runbook
- run_diagnostic
- summarize_incident

Also defines typed resource fixtures for the following memory URIs:

- memory://alerts/latest
- memory://runbooks/index
- memory://deltas/recent

---

### telemetry.py
Implements:

- Correlation IDs
- Budget object (tokens, milliseconds, cost)
- Telemetry event dataclass
- JSONL telemetry logger
- Timing helper

---

### mcp_tool_harness_server.py
Minimal MCP server implementing:

- initialize
- getResource
- callTool

Exposes three tools and logs full telemetry into JSONL files.

---

### mcp_tool_harness_client.py
Simple orchestrator that performs one OPAL loop:

1. Observe
2. Plan
3. Act
4. Learn

All events logged under `samples/client_telemetry.log`.

---

## Running the Warm-Up

Terminal 1:

```
cd 01_tool_harness
python mcp_tool_harness_server.py
```

Terminal 2:

```
python mcp_tool_harness_client.py
```

---

## Purpose of the Warm-Up

Prepares the environment for the full Incident Command agent by establishing:

- Tool schemas
- Resource access
- OPAL loop
- Structured telemetry
- Replay-ready logs
