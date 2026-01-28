# Troubleshooting Guide

This page focuses on issues that have already happened in this rewrite.

## Connection Issues

### Symptom: rapid connect/disconnect loop on the server

Common causes:

- Viewer is still running as a TCP server instead of a client
- You launched an old viewer binary
- Host/port mismatch

Checks:

- Server logs should show a stable session id, not many quick sessions
- Viewer logs should say "connected to AI server" (client mode)
- Confirm viewer connects to `127.0.0.1:9999`

## Viewer Freezes / Deadlocks

### Symptom: viewer freezes when disabling AI

Root causes tend to be:

- blocking socket operations on the UI thread
- join/wait patterns during shutdown

Current server behavior is non-blocking; focus on viewer shutdown paths.

## Status Channel Not Working

### Symptom: status messages go to local chat (channel 0)

Required viewer behavior:

1) Parse `parameters` in `chat_response`
2) Honor `parameters.channel` in the `CHAT` / `EMOTE` handlers

See: `docs/VIEWER_INTEGRATION.md`

## "LengthFinishReasonError" / parse failures

Symptom:

- model hits `max_tokens` (500) before completing structured output
- parse fails because output is truncated

Options:

- reduce prompt size (context window, experiences, environment volume)
- add stricter "be concise" instructions
- implement a retry-with-shorter-context strategy

## AI Replies to Itself

Protections exist server-side, but verify:

- the viewer is not echoing AI output back as a user message
- `from_name` / `logged_in_agent` are set correctly
- the AI's own avatar is not included in nearby agents

## Experiences Not Showing Up

Experiences are episodic now, not per message.

Triggers:

- `episode_summary.max_messages`
- `episode_summary.inactivity_seconds`
- `episode_summary.forced_interval_seconds`

Quick test:

- temporarily set `min_messages` low and `max_messages` small

## State Not Persisting Across Reconnects

State persistence depends on:

- `episode_summary.persist_state: true`
- stable persona + user_id

Important:

- state is keyed by persona + user_id
- changing either creates a new state file

## Neo4j Database Name Errors

Neo4j database names cannot contain underscores.

Example:

- Use `gpt5-roleplay` not `gpt5_roleplay`

## Compile Error: `shutdown(...)` overload mismatch

Symptom:

- compiler tries to call `AIProcessBridge::shutdown(...)` with socket args

Fix:

- call the global function explicitly: `::shutdown(...)`

