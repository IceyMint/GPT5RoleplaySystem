# GPT5RoleplaySystem Docs Index

Use this as the entry point when you need to re-orient quickly.

## Quick Start

- Run server: `python -m gpt5_roleplay_system.server --host 0.0.0.0 --port 9999 --config /path/to/config.yaml`
- Run tests: `python -m pytest -q`
- All-parts script: `python scripts/test_all_parts.py`

## Architecture + Flow

- Architecture overview: `docs/ARCHITECTURE.md`
- Protocol and message schemas: `docs/PROTOCOL.md`
- Config reference: `docs/CONFIG_REFERENCE.md`

## Memory + Episodes

- Memory model and episodic experiences: `docs/MEMORY_AND_EPISODES.md`
- Session state persistence across reconnects: `docs/STATE_PERSISTENCE.md`
- Neo4j fresh DB setup and constraints/indexes: `docs/neo4j_fresh_db.md`

## Viewer Integration

- Viewer integration contract and required changes: `docs/VIEWER_INTEGRATION.md`

## Observability + Debugging

- W&B/Weave logging and event names: `docs/OBSERVABILITY.md`
- Common issues and fixes: `docs/TROUBLESHOOTING.md`
- Developer workflows and "where to look" map: `docs/DEVELOPMENT_WORKFLOWS.md`

