import pytest
import time
import math
import asyncio
from gpt5_roleplay_system.pipeline import MessagePipeline

def test_record_latency_p95_calculation():
    # Setup dummy tracer
    class DummyTracer:
        def __init__(self):
            self.events = []
        def log_event(self, name, payload):
            self.events.append((name, payload))

    # Test p95 for exactly 100 samples
    pipeline = MessagePipeline.__new__(MessagePipeline)
    pipeline._tracer = DummyTracer()
    pipeline._latency_history = []
    pipeline._last_latency_summary_ts = time.time()

    # Pre-fill with 99 values
    pipeline._latency_history = [float(i) for i in range(99)]

    # Adding the 100th value should trigger the summary
    pipeline._record_latency(99.0, 1)

    # 0 to 99 are the 100 elements.
    # length = 100. 0.95 * 100 = 95. ceil(95) = 95. idx = max(0, 95 - 1) = 94.
    # history[94] = 94.0

    summary_events = [e for e in pipeline._tracer.events if e[0] == "perf_summary"]
    assert len(summary_events) == 1

    payload = summary_events[0][1]
    assert payload["count"] == 100
    assert payload["min_ms"] == 0.0
    assert payload["max_ms"] == 99.0
    assert payload["p95_ms"] == 94.0

    # Let's test again for a non-integer count like 101 to check math.ceil
    pipeline._latency_history = [float(i) for i in range(100)]

    # Add one more (making 101) but let's artificially set the timestamp way back so it triggers
    pipeline._last_latency_summary_ts = time.time() - 400.0
    pipeline._record_latency(100.0, 1)

    summary_events = [e for e in pipeline._tracer.events if e[0] == "perf_summary"]
    assert len(summary_events) == 2

    payload = summary_events[1][1]
    assert payload["count"] == 101
    assert payload["min_ms"] == 0.0
    assert payload["max_ms"] == 100.0
    # count = 101. 101 * 0.95 = 95.95. ceil(95.95) = 96. idx = 95.
    # history[95] = 95.0
    assert payload["p95_ms"] == 95.0
