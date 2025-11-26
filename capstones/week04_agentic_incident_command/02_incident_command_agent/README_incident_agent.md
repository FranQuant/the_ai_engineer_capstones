# Incident Command Agent (Scaffold)

Clean architecture-aligned scaffold following the warm-up harness principles:
- OPAL loop orchestrator (`incident_agent.py`)
- Memory store for `memory://` resources (`incident_memory.py`)
- Tool/resource schemas (`incident_schemas.py`)
- Planner stub (`incident_planner.py`)
- MCP server skeleton (`mcp_server.py`)
- Telemetry stubs with correlation/loop IDs (`telemetry.py`)
- Config stub (`config.yaml`)

TODO:
- Implement schemas, planner logic, and memory persistence.
- Wire MCP server, telemetry logging, and OPAL loop execution.
- Add samples/replay once implementations are complete.
