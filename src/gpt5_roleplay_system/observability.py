from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import weave
except ImportError:  # pragma: no cover - optional dependency
    weave = None


class Tracer:
    def start_run(self, run_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def log_event(self, name: str, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError


class NoOpTracer(Tracer):
    def start_run(self, run_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        return None

    def log_event(self, name: str, payload: Dict[str, Any]) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbTracer(Tracer):
    def __init__(self, project: str = "gpt5-roleplay") -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb is not installed") from exc
        self._wandb = wandb
        self._weave = weave
        self._project = project
        self._run = None

    def start_run(self, run_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        if self._run is None:
            if self._weave is not None:
                self._weave.init(self._project)
            self._run = self._wandb.init(project=self._project, name=run_name, config=config or {})

    def log_event(self, name: str, payload: Dict[str, Any]) -> None:
        if self._run is None:
            return
        data = {f"event/{name}": payload}
        self._wandb.log(data)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
