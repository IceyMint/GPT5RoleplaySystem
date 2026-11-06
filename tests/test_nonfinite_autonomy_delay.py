import math
from types import SimpleNamespace

from gpt5_roleplay_system.llm import _bundle_from_structured
from gpt5_roleplay_system.pipeline import MessagePipeline


def _parsed_with_delay(delay):
    return SimpleNamespace(
        actions=[],
        facts=[],
        participant_hints=[],
        summary_update=None,
        mood=None,
        status=None,
        autonomy_decision="wait",
        next_delay_seconds=delay,
    )


def test_structured_bundle_rejects_infinite_autonomy_delay():
    """A model-emitted Infinity must not escape the structured-output boundary.

    Production emitted next_delay_seconds=inf, which was accepted as a positive
    float and later crashed the session worker at an integer-conversion site.
    This exercises the LIVE mapping path (llm._bundle_from_structured); the
    former ResponseMapper copy it used to test was never called in production.
    """
    bundle = _bundle_from_structured(_parsed_with_delay(math.inf))

    assert bundle.next_delay_seconds is None


def test_structured_bundle_rejects_nan_autonomy_delay():
    bundle = _bundle_from_structured(_parsed_with_delay(math.nan))

    assert bundle.next_delay_seconds is None


def test_pipeline_rejects_infinite_autonomy_delay_from_custom_client():
    assert MessagePipeline._sanitize_autonomy_delay_hint(math.inf) is None
