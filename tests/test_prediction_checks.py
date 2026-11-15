"""Verifier unit tests; no predictive-accuracy or real-data measurements."""

import pytest
from pydantic import ValidationError

from causalbridge.discovery.schemas import DirectionPrediction
from causalbridge.verifier.checks import check_prediction


def method_claim(direction="A_to_B"):
    return dict(type="method_direction", evidence_ref="method_result", direction=direction)


@pytest.mark.parametrize("index", range(4))
def test_valid_packages_release_saved_answer(bound_direction, index):
    e, p, context = bound_direction(index)
    result = check_prediction(e, p, context=context)
    assert result.status == "pass" and result.release
    assert result.reason_codes == ("evidence_consistent",)
    assert result.checker_version == "direction-checker-0.3"
    assert result.proposal_digest == p.proposal_digest


def test_display_and_method_samples_are_independent(bound_direction):
    e, p, context = bound_direction()
    assert e.display_sample_digest != context.method_request.sample_digest
    assert len(e.display_rows) < context.method_request.selected_row_count
    assert check_prediction(e, p, context=context).release


def test_qualified_disagreement_is_allowed(bound_direction):
    e, p, context = bound_direction(direction="B_to_A", claims=[method_claim()])
    before = p.model_dump_json()
    result = check_prediction(e, p, context=context)
    assert result.release and result.claim_findings[0].status == "pass"
    assert before == p.model_dump_json() and p.predicted_direction == "B_to_A"


def test_row_and_summary_claims_match_saved_evidence(bound_direction):
    claims = [
        dict(type="display_row", evidence_ref="display_rows", row_index=0, values=[205.0, 9.7]),
        dict(type="summary", evidence_ref="summary", statistic="mean", value=[125.5, 8.95]),
        dict(type="summary", evidence_ref="summary", statistic="n_display", value=2.0),
    ]
    e, p, context = bound_direction(2, claims=claims)
    result = check_prediction(e, p, context=context)
    assert result.release
    assert [f.reason_code for f in result.claim_findings] == [
        "row_claim_consistent",
        "summary_claim_consistent",
        "summary_claim_consistent",
    ]


@pytest.mark.parametrize(
    "claim,code",
    [
        (
            dict(type="display_row", evidence_ref="display_rows", row_index=2, values=[0.0, 0.0]),
            "row_unavailable",
        ),
        (
            dict(type="display_row", evidence_ref="display_rows", row_index=0, values=[205.0, 9.8]),
            "row_value_mismatch",
        ),
        (
            dict(type="summary", evidence_ref="summary", statistic="std", value=[1.0, 1.0]),
            "statistic_unavailable",
        ),
        (
            dict(type="summary", evidence_ref="summary", statistic="mean", value=125.5),
            "summary_value_mismatch",
        ),
        (
            dict(type="summary", evidence_ref="summary", statistic="n_display", value=2.00001),
            "summary_value_mismatch",
        ),
        (method_claim(), "method_unavailable"),
    ],
)
def test_unverifiable_and_false_claims_fail(bound_direction, claim, code):
    e, p, context = bound_direction(2, claims=[claim])
    result = check_prediction(e, p, context=context)
    assert result.status == "fail" and not result.release
    assert code in result.reason_codes
    assert result.claim_findings[0].reason_code == code


@pytest.mark.parametrize(
    "claim,code",
    [
        (
            dict(type="display_row", evidence_ref="display_rows", row_index=0, values=[205.0, 9.7]),
            "rows_unavailable",
        ),
        (
            dict(type="summary", evidence_ref="summary", statistic="mean", value=[125.5, 8.95]),
            "summary_unavailable",
        ),
        (method_claim(), "method_unavailable"),
    ],
)
def test_names_only_cannot_cite_numeric_evidence(bound_direction, claim, code):
    e, p, context = bound_direction(0, claims=[claim])
    assert check_prediction(e, p, context=context).claim_findings[0].reason_code == code


@pytest.mark.parametrize(
    "index,direction,code",
    [
        (3, "B_to_A", "method_direction_mismatch"),
        (3, None, "method_direction_mismatch"),
        (4, "A_to_B", "method_direction_mismatch"),
        (5, "A_to_B", "method_threshold_unconfigured"),
        (5, None, "method_threshold_unconfigured"),
        (6, "A_to_B", "method_execution_error"),
        (6, None, "method_execution_error"),
    ],
)
def test_method_state_and_attribution(bound_direction, index, direction, code):
    e, p, context = bound_direction(index, claims=[method_claim(direction)])
    result = check_prediction(e, p, context=context)
    assert result.status == "fail" and code in result.reason_codes


def test_method_ambiguity_can_be_cited_as_abstention(bound_direction):
    e, p, context = bound_direction(4, claims=[method_claim(None)])
    assert check_prediction(e, p, context=context).release


@pytest.mark.parametrize("index", [4, 5, 6])
def test_method_nondecision_does_not_force_llm_abstention(bound_direction, index):
    e, p, context = bound_direction(index, claims=[])
    assert check_prediction(e, p, context=context).release


def test_saved_model_abstention_passes_but_never_releases(bound_direction):
    e, p, context = bound_direction(direction=None, abstained=True)
    result = check_prediction(e, p, context=context)
    assert result.status == "pass" and not result.release
    assert result.reason_codes == ("prediction_abstained",)


def test_no_context_is_explicit_checker_abstention(bound_direction):
    e, p, _ = bound_direction()
    result = check_prediction(e, p)
    assert result.status == "abstain" and not result.release
    assert result.reason_codes == ("verification_context_missing", "claim_binding_unverified")
    assert result.claim_findings[0].status == "abstain"


@pytest.mark.parametrize(
    "field,code",
    [
        ("method_request", "method_request_missing"),
        ("method_result_digest", "method_result_binding_missing"),
    ],
)
def test_missing_method_trust_input_abstains(bound_direction, field, code):
    e, p, context = bound_direction()
    context = context.model_copy(update={field: None})
    result = check_prediction(e, p, context=context)
    assert result.status == "abstain" and code in result.reason_codes
    assert not result.release


def test_checker_is_pure_and_never_calls_model_or_anm(bound_direction, monkeypatch):
    import socket

    from causalbridge.llm.client import OllamaModel

    def forbidden(*args, **kwargs):
        pytest.fail("Checker tried to execute external work")

    monkeypatch.setattr(OllamaModel, "generate", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    e, p, context = bound_direction()
    before = [obj.model_dump_json() for obj in (e, p, context)]
    # The checker has no ANM callable or resolver argument. Importing one is also
    # forbidden, even if the teammate module is added later.
    import sys
    import types

    anm = types.ModuleType("causalbridge.discovery.anm")
    anm.run = anm.execute = anm.fit = forbidden
    monkeypatch.setitem(sys.modules, "causalbridge.discovery.anm", anm)
    a = check_prediction(e, p, context=context)
    b = check_prediction(e, p, context=context)
    assert a == b and a.release
    assert before == [obj.model_dump_json() for obj in (e, p, context)]


@pytest.mark.parametrize(
    "extra",
    [
        {"probability": 0.9},
        {"confidence": 0.9},
        {"score_ab": 0.9},
    ],
)
def test_numerical_probability_overclaims_remain_schema_errors(bound_direction, extra):
    _, p, _ = bound_direction()
    payload = p.model_dump(mode="json")
    payload["claims"][0].update(extra)
    with pytest.raises(ValidationError):
        DirectionPrediction.model_validate(payload)


@pytest.mark.parametrize("reference", ["hidden_rows", "method_result", "gold_direction"])
def test_invalid_reference_returns_explicit_finding(bound_direction, reference):
    e, p, context = bound_direction(2, claims=[])
    # Bypass parsing to prove the checker revalidates Pydantic model_copy data.
    p = p.model_copy(
        update={
            "claims": (
                dict(type="display_row", evidence_ref=reference, row_index=0, values=[205.0, 9.7]),
            )
        }
    )
    result = check_prediction(e, p, context=context)
    assert result.status == "fail"
    assert result.reason_codes == ("invalid_prediction",)
    assert result.claim_findings[0].reason_code == "claim_schema_invalid"


def test_adversarial_review_claim_findings_do_not_depend_on_order(bound_direction):
    good = dict(type="summary", evidence_ref="summary", statistic="n_display", value=2.0)
    bad = dict(type="display_row", evidence_ref="display_rows", row_index=0, values=[0.0, 0.0])
    for claims in ([bad, good], [good, bad]):
        e, p, context = bound_direction(2, claims=claims)
        result = check_prediction(e, p, context=context)
        assert result.status == "fail"
        finding = result.claim_findings[claims.index(good)]
        assert finding.status == "pass" and finding.reason_code == "summary_claim_consistent"


@pytest.mark.parametrize("location", ["prediction", "claim", "evidence", "method"])
def test_adversarial_review_hidden_model_copy_extras_are_not_dropped(bound_direction, location):
    e, p, context = bound_direction()
    if location == "prediction":
        p = p.model_copy(update={"probability": 0.95})
    elif location == "claim":
        p = p.model_copy(update={"claims": (p.claims[0].model_copy(update={"probability": 0.95}),)})
    elif location == "evidence":
        e = e.model_copy(update={"reference_direction": "A_to_B"})
    else:
        e = e.model_copy(
            update={"method_result": e.method_result.model_copy(update={"probability": 0.95})}
        )
    result = check_prediction(e, p, context=context)
    assert result.status == "fail" and not result.release
    assert result.reason_codes == (
        "invalid_prediction" if location in ("prediction", "claim") else "invalid_evidence",
    )


def test_row_claim_remains_checkable_when_method_request_is_missing(bound_direction):
    e, p, context = bound_direction(
        claims=[
            dict(type="display_row", evidence_ref="display_rows", row_index=0, values=[205.0, 9.7])
        ]
    )
    context = context.model_copy(update={"method_request": None})
    result = check_prediction(e, p, context=context)
    assert result.status == "abstain" and not result.release
    assert result.claim_findings[0].status == "pass"


def test_summary_claims_cover_fixed_vocabulary(bound_direction):
    from causalbridge.discovery.schemas import DirectionEvidence, canonical_digest

    e, p, context = bound_direction(2, claims=[])
    summary = dict(
        n_display=2,
        mean=[125.5, 8.95],
        std=[1.0, 2.0],
        minimum=[46.0, 8.2],
        maximum=[205.0, 9.7],
        correlation=0.5,
    )
    e = DirectionEvidence.model_validate(e.model_dump(mode="json") | {"summary": summary})
    claims = [
        dict(type="summary", evidence_ref="summary", statistic=key, value=value)
        for key, value in summary.items()
    ]
    p = DirectionPrediction.model_validate(
        p.model_dump(mode="json") | {"package_digest": e.package_digest, "claims": claims}
    )
    context = context.model_copy(
        update={"package_digest": e.package_digest, "proposal_digest": p.proposal_digest}
    )
    assert canonical_digest(e.model_dump(mode="json")) == e.package_digest
    assert check_prediction(e, p, context=context).release


@pytest.mark.parametrize("field", ["gold_direction", "reference_direction", "evaluator_metadata"])
def test_verification_inputs_reject_evaluator_fields(bound_direction, field):
    from causalbridge.verifier.bindings import ANMRequestBinding, VerificationContext

    _, _, context = bound_direction()
    with pytest.raises(ValidationError):
        VerificationContext.model_validate(context.model_dump(mode="json") | {field: "secret"})
    with pytest.raises(ValidationError):
        ANMRequestBinding.model_validate(
            context.method_request.model_dump(mode="json") | {field: "secret"}
        )


def test_unchanged_public_prediction_schema_rejects_unknown_claim_types(bound_direction):
    _, p, _ = bound_direction()
    for claim in [
        dict(type="causal_probability", value=0.9),
        dict(type="method_score", evidence_ref="method_result", score=0.9),
    ]:
        with pytest.raises(ValidationError):
            DirectionPrediction.model_validate(p.model_dump(mode="json") | {"claims": [claim]})


def test_saved_phase_two_examples_reproduce_exact_checks():
    import json
    from pathlib import Path

    from causalbridge.discovery.schemas import DirectionEvidence
    from causalbridge.verifier.bindings import VerificationContext
    from causalbridge.verifier.checks import PredictionCheck

    source = Path(__file__).parents[1] / "examples/track_b_phase2.jsonl"
    for line in source.read_text().splitlines():
        record = json.loads(line)
        e = DirectionEvidence.model_validate(record["evidence"])
        p = DirectionPrediction.model_validate(record["saved_prediction"])
        context = VerificationContext.model_validate(record["context"])
        expected = PredictionCheck.model_validate(record["check"])
        actual = check_prediction(e, p, context=context)
        # Historical fixtures retain their original checker identity and decisions.
        assert actual.model_dump(exclude={"checker_version"}) == expected.model_dump(
            exclude={"checker_version"}
        )
        assert actual.checker_version == "direction-checker-0.3"


@pytest.mark.parametrize(
    "score_ab,score_ba,expected",
    [
        (0.0, 0.25, None),
        (0.25, 0.0, None),
        (0.0, 0.5, "A_to_B"),
        (0.5, 0.0, "B_to_A"),
    ],
)
def test_checker_threshold_boundaries_in_both_directions(
    bound_direction,
    score_ab,
    score_ba,
    expected,
):
    from causalbridge.discovery.schemas import DirectionEvidence, canonical_digest

    e, p, context = bound_direction(claims=[])
    data = e.model_dump(mode="json")
    data["method_result"].update(
        score_ab=score_ab,
        score_ba=score_ba,
        margin=score_ba - score_ab,
        threshold=0.25,
        decision="abstain" if expected is None else expected,
        predicted_direction=expected,
        reason_code="ambiguous_margin" if expected is None else None,
    )
    e = DirectionEvidence.model_validate(data)
    p = DirectionPrediction.model_validate(
        p.model_dump(mode="json")
        | dict(package_digest=e.package_digest, claims=[method_claim(expected)])
    )
    # These values represent a newly captured original fixture, not a corruption.
    context = context.model_copy(
        update=dict(
            package_digest=e.package_digest,
            proposal_digest=p.proposal_digest,
            method_result_digest=canonical_digest(e.method_result.model_dump(mode="json")),
            method_request=context.method_request.model_copy(update={"threshold": 0.25}),
        )
    )
    result = check_prediction(e, p, context=context)
    assert result.release and result.claim_findings[0].status == "pass"
