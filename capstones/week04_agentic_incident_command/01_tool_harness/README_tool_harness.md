<table width="100%">
  <tr>
    <td style="vertical-align: top;">
      <h1>MCP Tool Harness (Warm-Up)</h1>
      <p>
        This module provides a minimal MCP server and client used to explore the
        core mechanics of the OPAL loop (Observe → Plan → Act → Learn) before
        building the full Incident Command Agent in <code>02_incident_command_agent/</code>.
      </p>
    </td>
    <td align="right" width="200">
      <img src="../../assets/tae_logo.png" alt="TAE Banner" width="160">
    </td>
  </tr>
</table>




## Components
- **mcp_tool_harness_server.py** — exposes 3 tools and 3 memory:// resources  
- **mcp_tool_harness_client.py** — runs a single deterministic OPAL loop  
- **schemas.py** — JSON schemas for tools and resources  
- **telemetry.py** — structured telemetry + JSONL logging  
- **samples/** — example server/client logs  

## How to Run

**Terminal 1 (server):**
```bash
cd 01_tool_harness
python mcp_tool_harness_server.py
```
**Terminal 2 (client):**

```bash
python mcp_tool_harness_client.py
```

Logs will appear under `samples/`.


## Purpose

This warm-up demonstrates:

- MCP message flow

- Tool invocation

- Resource retrieval

- Telemetry + replay-ready traces

These foundations are used directly in **Module 02**.
