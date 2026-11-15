"""Single-call Track B predictor using the existing transport and strict parser."""

from copy import deepcopy
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import Field

from causalbridge.discovery.schemas import (
    Digest,
    DirectionEvidence,
    DirectionPrediction,
    PromptCondition,
    canonical_digest,
)
from causalbridge.llm.client import Generation, GenerationFailure, JSONModel, ModelConfig
from causalbridge.llm.parser import messages, parse_json
from causalbridge.schemas import Record

QUESTION = "Which direction is more plausible, A → B or B → A? You may abstain."
PROMPT_VERSION = "direction-0.1"


def prompt_version_for(prompt_condition: PromptCondition) -> str:
    return (
        "direction-0.2-compact"
        if prompt_condition.version == "integrated-track-b-v2-compact"
        else PROMPT_VERSION
    )


class PredictionExecution(Record):
    schema_version: Literal["0.1"] = "0.1"
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    prompt_version: str = PROMPT_VERSION
    evidence: DirectionEvidence
    package_digest: Digest
    display_sample_digest: Digest | None
    prompt_condition: PromptCondition
    prompt_digest: Digest
    request_messages: tuple[dict[str, str], ...]
    output_schema: dict
    generation_config: ModelConfig | None
    generation: Generation | None = None
    prediction: DirectionPrediction | None = None
    failure_stage: Literal["generate", "parse", "binding"] | None = None
    error: str | None = None
    latency_seconds: float = Field(ge=0)


def direction_messages(
    evidence: DirectionEvidence, prompt_condition: PromptCondition
) -> list[dict[str, str]]:
    """Pure rendering hook for runner cache keys; identical to predictor messages."""
    return messages(
        prompt_condition.instructions + "\nReturn JSON only matching the supplied schema. "
        "Use A_to_B, B_to_A or null with abstained=true and a reason. Copy the supplied "
        "case_id and package_digest. Scores and margins are diagnostics, not calibrated causal "
        "probabilities. Observations and ANM preferences are not proof of causation. "
        "Claims may cite only supplied evidence. Qualified disagreement with ANM is permitted.",
        dict(
            question=QUESTION,
            evidence=evidence.model_dump(mode="json"),
            case_id=evidence.case_id,
            package_digest=evidence.package_digest,
            display_sample_digest=evidence.display_sample_digest,
        ),
    )


def predict_direction(
    evidence: DirectionEvidence,
    prompt_condition: PromptCondition,
    model_client: JSONModel,
) -> PredictionExecution:
    """Generate once; never call ANM, score labels, retry, or choose a condition.

    Parsed but misbound predictions remain in the trace with failure_stage=binding.
    Callers must check failure_stage before using a prediction.
    """
    started = perf_counter()
    evidence = DirectionEvidence.model_validate_json(evidence.model_dump_json())
    prompt_condition = PromptCondition.model_validate_json(prompt_condition.model_dump_json())
    prompt = direction_messages(evidence, prompt_condition)
    schema = DirectionPrediction.model_json_schema()
    # Keep the exact request for audit even if a custom transport mutates its arguments.
    saved_prompt = tuple(dict(m) for m in prompt)
    saved_schema = deepcopy(schema)
    generation = prediction = error = None
    stage = "generate"
    try:
        generation = model_client.generate(prompt, schema)
        if generation.status != "complete":
            raise GenerationFailure(generation)
        stage = "parse"
        prediction = parse_json(generation.text, DirectionPrediction)
        stage = "binding"
        if (
            prediction.case_id != evidence.case_id
            or prediction.package_digest != evidence.package_digest
        ):
            raise ValueError("Prediction case/package binding mismatch")
        stage = None
    except GenerationFailure as exc:
        generation = exc.generation
        error = f"GenerationFailure: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return PredictionExecution(
        prompt_version=prompt_version_for(prompt_condition),
        evidence=evidence,
        package_digest=evidence.package_digest,
        display_sample_digest=evidence.display_sample_digest,
        prompt_condition=prompt_condition,
        prompt_digest=canonical_digest(saved_prompt),
        request_messages=saved_prompt,
        output_schema=saved_schema,
        generation_config=getattr(model_client, "config", None),
        generation=generation,
        prediction=prediction,
        failure_stage=stage,
        error=error,
        latency_seconds=perf_counter() - started,
    )
