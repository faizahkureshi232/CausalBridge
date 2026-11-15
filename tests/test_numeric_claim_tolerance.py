"""Checker 0.3 numeric policy controls; no live inference or cached artifact mutation."""

import math

import pytest

from causalbridge.discovery.schemas import DirectionEvidence, DirectionPrediction
from causalbridge.verifier.checks import check_prediction


def bind_numeric(bound_direction, *, expected, actual, kind="summary", statistic="mean"):
    e, p, context = bound_direction(2, claims=[])
    data = e.model_dump(mode="json")
    if kind == "summary":
        data["summary"][statistic] = expected
        claim = dict(type="summary", evidence_ref="summary", statistic=statistic, value=actual)
    else:
        data["display_rows"][0] = expected
        claim = dict(type="display_row", evidence_ref="display_rows", row_index=0, values=actual)
    e = DirectionEvidence.model_validate(data)
    p = DirectionPrediction.model_validate(
        p.model_dump(mode="json")
        | dict(
            package_digest=e.package_digest,
            claims=[claim],
        )
    )
    context = context.model_copy(
        update=dict(
            package_digest=e.package_digest,
            proposal_digest=p.proposal_digest,
            display_sample_digest=e.display_sample_digest,
        )
    )
    return e, p, context


@pytest.mark.parametrize(
    "kind,expected,actual",
    [
        (
            "summary",
            [-0.18179807460923827, 0.9974522491495209],
            [-0.1817980746092382, 0.9974522491495209],
        ),
        (
            "row",
            [0.3469462416659086, -0.12044657480477401],
            [0.3469462416659086, -0.1204465748047740],
        ),
    ],
)
def test_both_observed_smoke_rounding_mismatches(bound_direction, kind, expected, actual):
    e, p, context = bind_numeric(bound_direction, expected=expected, actual=actual, kind=kind)
    before = [obj.model_dump_json() for obj in (e, p, context)]
    result = check_prediction(e, p, context=context)
    assert result.status == "pass" and result.release
    assert result.claim_findings[0].status == "pass"
    assert before == [obj.model_dump_json() for obj in (e, p, context)]


@pytest.mark.parametrize("kind", ["summary", "row"])
@pytest.mark.parametrize(
    "expected,actual,accepted",
    [
        ([1.0, -2.0], [1.0, -2.0], True),  # Exact equality.
        ([0.0, 0.0], [5e-13, -5e-13], True),  # Near zero, both signs.
        ([0.0, 0.0], [math.nextafter(1e-12, 0.0), 0.0], True),
        ([0.0, 0.0], [math.nextafter(1e-12, math.inf), 0.0], False),
        ([2.0, -3.0], [math.nextafter(2.0 + 2e-12, 2.0), -3.0], True),
        ([2.0, -3.0], [math.nextafter(2.0 + 2e-12, math.inf), -3.0], False),
        ([-10.0, -2.0], [-10.0 + 5e-12, -2.0], True),  # Relative tolerance.
        ([-10.0, -2.0], [-10.0 + 2e-11, -2.0], False),
        ([1e6, -1e6], [1e6 + 5e-7, -1e6], True),
        ([1e6, -1e6], [1e6 + 2e-6, -1e6], False),
        ([1.0, 2.0], [2.0, 1.0], False),  # A/B order is exact.
        ([1.0, 2.0], [1.001, 2.0], False),  # Material change.
    ],
)
def test_pair_policy_boundaries(bound_direction, kind, expected, actual, accepted):
    e, p, context = bind_numeric(bound_direction, expected=expected, actual=actual, kind=kind)
    result = check_prediction(e, p, context=context)
    assert result.release is accepted
    assert result.status == ("pass" if accepted else "fail")


@pytest.mark.parametrize(
    "value,accepted",
    [
        (0.5, True),
        (0.5 + 5e-13, True),
        (0.5 + 2e-12, False),
    ],
)
def test_scalar_summary_tolerance(bound_direction, value, accepted):
    e, p, context = bind_numeric(
        bound_direction, expected=0.5, actual=value, statistic="correlation"
    )
    assert check_prediction(e, p, context=context).release is accepted


def test_discrete_count_is_exact(bound_direction):
    for actual, accepted in [(2.0, True), (2.0 + 5e-13, False)]:
        e, p, context = bind_numeric(
            bound_direction, expected=2, actual=actual, statistic="n_display"
        )
        assert check_prediction(e, p, context=context).release is accepted


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("wrong_field", "summary_value_mismatch"),
        ("wrong_reference", "invalid_prediction"),
        ("missing_statistic", "statistic_unavailable"),
        ("wrong_row", "row_value_mismatch"),
        ("missing_row", "row_unavailable"),
        ("wrong_shape", "summary_value_mismatch"),
    ],
)
def test_field_and_citation_binding_precedes_tolerance(bound_direction, fault, reason):
    kind = "row" if fault in {"wrong_row", "missing_row"} else "summary"
    e, p, context = bind_numeric(
        bound_direction, expected=[1.0, 2.0], actual=[1.0 + 5e-13, 2.0], kind=kind
    )
    claim = p.claims[0].model_dump(mode="json")
    if fault == "wrong_field":
        claim["statistic"] = "n_display"
    elif fault == "wrong_reference":
        claim["evidence_ref"] = "method_result"
    elif fault == "missing_statistic":
        claim["statistic"] = "std"
    elif fault == "wrong_row":
        claim["row_index"] = 1
    elif fault == "missing_row":
        claim["row_index"] = 999
    elif fault == "wrong_shape":
        claim["value"] = 1.0
    if fault == "wrong_reference":
        p = p.model_copy(update={"claims": (claim,)})
    else:
        p = DirectionPrediction.model_validate(p.model_dump(mode="json") | {"claims": [claim]})
        context = context.model_copy(update={"proposal_digest": p.proposal_digest})
    result = check_prediction(e, p, context=context)
    assert result.status == "fail" and not result.release
    assert reason in result.reason_codes


@pytest.mark.parametrize(
    "field,new_value,reason",
    [
        ("case_id", "wrong-case", "case_id_mismatch"),
        ("package_digest", "sha256:" + "f" * 64, "package_digest_mismatch"),
        ("proposal_digest", "sha256:" + "e" * 64, "proposal_digest_mismatch"),
        ("display_sample_digest", "sha256:" + "a" * 64, "display_sample_mismatch"),
        ("ab_mapping_id", "wrong-mapping", "ab_mapping_mismatch"),
        ("mapping_seed", 262, "ab_mapping_mismatch"),
    ],
)
def test_close_numbers_cannot_bypass_context_identity(
    bound_direction, monkeypatch, field, new_value, reason
):
    e, p, context = bind_numeric(bound_direction, expected=[1.0, 2.0], actual=[1.0 + 5e-13, 2.0])
    context = context.model_copy(update={field: new_value})
    # No isclose evaluation is permitted for an incorrectly bound citation.
    monkeypatch.setattr(
        "causalbridge.verifier.checks.math.isclose",
        lambda *a, **k: pytest.fail("Tolerance before binding"),
    )
    result = check_prediction(e, p, context=context)
    assert not result.release and result.status == "fail"
    assert reason in result.reason_codes


def test_close_numbers_without_receipt_are_not_certified(bound_direction, monkeypatch):
    e, p, _ = bind_numeric(bound_direction, expected=[1.0, 2.0], actual=[1.0 + 5e-13, 2.0])
    monkeypatch.setattr(
        "causalbridge.verifier.checks.math.isclose",
        lambda *a, **k: pytest.fail("Tolerance without receipt"),
    )
    assert not check_prediction(e, p).release


@pytest.mark.parametrize("field", ["case_id", "package_digest"])
def test_prediction_identity_remains_exact(bound_direction, field):
    e, p, context = bind_numeric(bound_direction, expected=[1.0, 2.0], actual=[1.0 + 5e-13, 2.0])
    value = "wrong-case" if field == "case_id" else "sha256:" + "b" * 64
    p = p.model_copy(update={field: value})
    result = check_prediction(e, p, context=context)
    assert result.status == "fail" and not result.release


@pytest.mark.parametrize(
    "field,new_value,reason",
    [
        ("sample_digest", "sha256:" + "c" * 64, "method_sample_mismatch"),
        ("config_digest", "sha256:" + "d" * 64, "method_config_mismatch"),
        ("case_id", "other-case", "method_case_mismatch"),
        ("threshold", 0.001 + 5e-16, "method_config_mismatch"),
    ],
)
def test_method_provenance_and_settings_remain_exact(bound_direction, field, new_value, reason):
    e, p, context = bound_direction()
    original = context.method_request
    if field == "threshold":
        new_value = original.threshold + 5e-16
    context = context.model_copy(
        update={
            "method_request": original.model_copy(update={field: new_value}),
        }
    )
    result = check_prediction(e, p, context=context)
    assert result.status == "fail" and reason in result.reason_codes


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None])
def test_nonfinite_and_missing_claim_numbers_are_rejected(bound_direction, value):
    e, p, context = bound_direction(2, claims=[])
    p = p.model_copy(
        update={
            "claims": (
                dict(
                    type="display_row",
                    evidence_ref="display_rows",
                    row_index=0,
                    values=[value, 9.7],
                ),
            )
        }
    )
    # Nonfinite bypasses may be rejected at canonical JSON's fail-closed boundary.
    try:
        result = check_prediction(e, p, context=context)
    except ValueError:
        assert value is not None and not math.isfinite(value)
    else:
        assert result.status == "fail" and not result.release
        assert "invalid_prediction" in result.reason_codes


def test_direction_and_reason_codes_not_normalized(bound_direction):
    e, p, context = bound_direction(direction="B_to_A", claims=[])
    result = check_prediction(e, p, context=context)
    assert result.release  # Disagreement without false method attribution stays allowed.
    assert p.predicted_direction == "B_to_A" and result.reason_codes == ("evidence_consistent",)
    p = p.model_copy(
        update={
            "claims": (
                dict(type="method_direction", evidence_ref="method_result", direction="B_to_A"),
            )
        }
    )
    result = check_prediction(e, p, context=context)
    assert not result.release and "method_direction_mismatch" in result.reason_codes
