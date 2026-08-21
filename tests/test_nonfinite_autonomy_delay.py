import math
from types import SimpleNamespace

from gpt5_roleplay_system.llm_response_mapper import ResponseMapper
from gpt5_roleplay_system.pipeline import MessagePipeline


def test_structured_bundle_rejects_infinite_autonomy_delay():
    """A model-emitted Infinity must not escape the structured-output boundary.

    Production emitted next_delay_seconds=inf, which was accepted as a positive
    float and later crashed the session worker at an integer-conversion site.
    """
    parsed = SimpleNamespace(
        actions=[],
        facts=[],
        participant_hints=[],
        summary_update=None,
        mood=None,
        status=None,
        autonomy_decision="wait",
        next_delay_seconds=math.inf,
    )

    bundle = ResponseMapper().bundle_from_structured(parsed)

    assert bundle.next_delay_seconds is None


def test_pipeline_rejects_infinite_autonomy_delay_from_custom_client():
    assert MessagePipeline._sanitize_autonomy_delay_hint(math.inf) is None
