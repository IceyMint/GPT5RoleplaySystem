# Observability (W&B + Weave)

Core tracer code:

- `src/gpt5_roleplay_system/observability.py`

The server uses a simple tracer interface. When `wandb.enabled: true`, it uses
`WandbTracer`:

- initializes W&B
- initializes Weave if installed
- logs structured events under `event/<name>`

## How to Enable

In `config.yaml`:

```yaml
wandb:
  enabled: true
  project: gpt5-roleplay
  run_name: null
```

Auth:

- set `WANDB_API_KEY` in the environment, or
- use `api_keys.wandb_api_key` in config

## Key Event Names

Most useful events come from the pipeline:

- `event/environment_update`
  - payload: `{agents: <count>}`
- `event/llm_address_check`
  - payload: sender, text, persona, participants, location
- `event/llm_address_result`
  - payload: sender, addressed (bool)
- `event/llm_prompt_bundle`
  - payload: full structured prompt context for chat mode
- `event/llm_response_bundle`
  - payload: structured model output (text/actions/facts/hints/summary)
- `event/llm_prompt_autonomy`
  - payload: structured prompt context for autonomy mode
- `event/llm_response_autonomy`
  - payload: structured autonomy output
- `event/episode_summary`
  - payload: reason, messages, overlap
- `event/response`
  - payload: number of actions + batch size

## Notes and Safety

- These events can include conversation text and environment details.
- Avoid enabling W&B on sensitive data without considering privacy.
- The tracer intentionally logs structured payloads to make debugging faster.

