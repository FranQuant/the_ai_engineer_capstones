# MCP Tool Harness (Warm-Up Module)

This directory implements the **Tool Harness warm-up project** for **Week-04 of *The AI Engineer*** program.  
It provides a compact, self-contained MCP server and client designed to exercise the foundational mechanisms behind modern agent systems:

- JSON Schema–driven tool interfaces  
- Deterministic **OPAL loops** (Observe → Plan → Act → Learn)  
- Fully typed telemetry instrumentation  
- Replay-ready execution traces (JSONL)  
- In-memory “`memory://`” resources  
- End-to-end local server/client interaction  

This warm-up module mirrors the core architecture used later in the **full Incident Command Agent** located in `02_incident_command_agent/`, but in a simplified, didactic form.

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

### **schemas.py**
Defines JSON schemas for the three warm-up tools:

- `retrieve_runbook`
- `run_diagnostic`
- `summarize_incident`

Also contains typed resource fixtures for the following `memory://` URIs:

- `memory://alerts/latest`
- `memory://runbooks/index`
- `memory://deltas/recent`

These schemas make the tool harness fully **MCP-compliant**, enabling deterministic input/output validation.

---

### **telemetry.py**
Implements core observability primitives:

- Correlation IDs  
- Token/time/cost **Budget** object  
- `TelemetryEvent` dataclass  
- High-resolution timing decorator  
- JSONL logger compatible with downstream replay tools  

Telemetry is structured and analyzable — ideal for debugging and benchmarking small agents.

---

### **mcp_tool_harness_server.py**
A minimal MCP server implementing:

- **initialize**  
- **getResource**  
- **callTool**

Exposes the three warm-up tools and logs server-side telemetry into `samples/server_telemetry.log`.

The server provides:

- Declarative tool registration  
- Typed resource retrieval  
- Deterministic execution  
- Fully local, dependency-free behavior  

---

### **mcp_tool_harness_client.py**
A lightweight orchestrator that performs one complete **OPAL loop**:

1. **Observe** → query server capabilities & resources  
2. **Plan** → simple policy-driven planner logic  
3. **Act** → execute MCP tools  
4. **Learn** → retrieve deltas/resources post-execution  

Client telemetry is logged into `samples/client_telemetry.log`.

This file demonstrates the exact interaction pattern used by more sophisticated agents later in Week-04.

---

## Running the Warm-Up

### Terminal 1 — Start the MCP server

```
cd 01_tool_harness
python mcp_tool_harness_server.py
```

### Terminal 2 — Run the sample client

```
python mcp_tool_harness_client.py
```

Both logs will appear in `samples/`.

---

## Example Output (Client)

```
=== OPAL Summary ===
{
  "observations": {...},
  "plan": [...],
  "results": [...],
  "learn": {...}
}
```

Full traces are available in:

- `samples/client_telemetry.log`
- `samples/server_telemetry.log`

---

## Purpose of the Warm-Up

This module prepares you for the full multi-step Incident Command Agent by establishing the foundations:

- Tool schemas  
- Resource access patterns  
- Deterministic OPAL loop  
- Structured telemetry  
- Replay-ready logging  
- Local server/client interplay  

Once these fundamentals are mastered, we are ready to build the full agent system in:

```
02_incident_command_agent/
```

---

## Version

**Week-04 — Module 01: MCP Tool Harness  
AI Engineer Program (Nov 2025 Cohort)**

