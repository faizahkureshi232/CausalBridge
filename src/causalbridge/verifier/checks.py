"""Deterministic Track B consistency gate; never establishes causal truth."""

import math
from typing import Literal

from pydantic import Field, StrictBool, ValidationError, model_validator

from causalbridge.discovery.schemas import (
    Digest,
    DirectionEvidence,
    DirectionPrediction,
    DisplayRowClaim,
    MethodDirectionClaim,
    canonical_digest,
)
from causalbridge.schemas import Record
from causalbridge.verifier.bindings import VerificationContext

NUMERIC_CLAIM_REL_TOL = 1e-12
NUMERIC_CLAIM_ABS_TOL = 1e-12


class ClaimFinding(Record):
    claim_index: int = Field(strict=True, ge=0)
    status: Literal["pass", "fail", "abstain"]
    reason_code: str = Field(min_length=1)


class PredictionCheck(Record):
    case_id: str
    package_digest: Digest
    proposal_digest: Digest
    checker_version: str = "direction-checker-0.3"
    status: Literal["pass", "fail", "abstain"]
    release: StrictBool
    reason_codes: tuple[str, ...] = Field(min_length=1)
    claim_findings: tuple[ClaimFinding, ...] = ()

    @model_validator(mode="after")
    def validate_release(self):
        if self.release and self.status != "pass":
            raise ValueError("Only a passing process check can release a saved answer")
        return self


def check_prediction(
    evidence: DirectionEvidence,
    saved_prediction: DirectionPrediction,
    *,
    context: VerificationContext | None = None,
) -> PredictionCheck:
    """Check exact cached S1 against original evidence and independent receipts.

    All contradictions fail. Missing trust records abstain. Otherwise pass;
    release only if S1 supplied a direction. No replacement direction is emitted.
    Revalidation catches model_copy/model_construct bypasses of frozen schemas.
    Malformed non-JSON objects and invalid context schemas raise at the API
    boundary; JSON-representable malformed evidence/proposals return fail.
    """
    raw_evidence = _payload(evidence)
    raw_prediction = _payload(saved_prediction)
    package_digest = canonical_digest(raw_evidence)
    proposal_digest = canonical_digest(raw_prediction)
    # Even the rejection path needs a valid diagnostic identity. Never coerce
    # an invalid ID into a real case ID or borrow one from another record.
    case_id = raw_evidence.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        case_id = "<invalid_case_id>"

    def finish(status, codes, findings=()):
        return PredictionCheck(
            case_id=case_id,
            package_digest=package_digest,
            proposal_digest=proposal_digest,
            status=status,
            release=status == "pass" and not saved_prediction.abstained,
            reason_codes=tuple(dict.fromkeys(codes)),
            claim_findings=tuple(findings),
        )

    try:
        evidence = DirectionEvidence.model_validate(raw_evidence)
    except ValidationError:
        return finish("fail", ("invalid_evidence",))
    try:
        saved_prediction = DirectionPrediction.model_validate(raw_prediction)
    except ValidationError as exc:
        indices = sorted(
            {
                err["loc"][1]
                for err in exc.errors()
                if len(err["loc"]) > 1
                and err["loc"][0] == "claims"
                and isinstance(err["loc"][1], int)
            }
        )
        return finish(
            "fail",
            ("invalid_prediction",),
            [
                ClaimFinding(claim_index=i, status="fail", reason_code="claim_schema_invalid")
                for i in indices
            ],
        )

    failures, unknowns = [], []
    if saved_prediction.case_id != evidence.case_id:
        failures.append("case_id_mismatch")
    if saved_prediction.package_digest != package_digest:
        failures.append("package_digest_mismatch")
    if context is None:
        unknowns.append("verification_context_missing")
    else:
        context = VerificationContext.model_validate(_payload(context))
        comparisons = (
            (context.case_id, evidence.case_id, "case_id_mismatch"),
            (context.package_digest, package_digest, "package_digest_mismatch"),
            (context.proposal_digest, proposal_digest, "proposal_digest_mismatch"),
            (context.ab_mapping_id, evidence.ab_mapping_id, "ab_mapping_mismatch"),
            (context.mapping_seed, evidence.mapping_seed, "ab_mapping_mismatch"),
            (
                context.display_sample_digest,
                evidence.display_sample_digest,
                "display_sample_mismatch",
            ),
        )
        failures.extend(code for expected, actual, code in comparisons if expected != actual)

    evidence_bound = context is not None and not failures
    method_failures, method_unknowns = _method_bindings(evidence, context)
    method_bound = evidence_bound and not method_failures and not method_unknowns
    failures.extend(method_failures)
    unknowns.extend(method_unknowns)
    findings = []
    for index, claim in enumerate(saved_prediction.claims):
        claim_bound = method_bound if isinstance(claim, MethodDirectionClaim) else evidence_bound
        status, code = _check_claim(evidence, claim, numeric_bound=claim_bound)
        if status == "pass" and not claim_bound:
            # A locally correct citation is not certified against an unbound cache.
            status, code = "abstain", "claim_binding_unverified"
        findings.append(ClaimFinding(claim_index=index, status=status, reason_code=code))
        if status == "fail":
            failures.append(code)
        elif status == "abstain":
            unknowns.append(code)
    if failures:
        return finish("fail", failures + unknowns, findings)
    if unknowns:
        return finish("abstain", unknowns, findings)
    return finish(
        "pass",
        ("prediction_abstained" if saved_prediction.abstained else "evidence_consistent",),
        findings,
    )


def _payload(value):
    """Snapshot even undeclared model_copy fields; model_dump silently drops them.

    Revalidation must see all supplied record fields, including nested records.
    Only runtime Record/JSON collections are traversed; no property/resolver calls.
    """
    if isinstance(value, Record):
        fields = vars(value) | (value.model_extra or {})
        return {key: _payload(item) for key, item in fields.items()}
    if isinstance(value, dict):
        return {key: _payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_payload(item) for item in value]
    return value


def _method_bindings(evidence, context):
    """Compare result to the saved request, never to display-sample identity."""
    failures, unknowns = [], []
    method = evidence.method_result
    if context is None:
        return failures, unknowns
    if method is None:
        if context.method_request is not None or context.method_result_digest is not None:
            failures.append("method_unavailable")
        return failures, unknowns
    if context.method_result_digest is None:
        unknowns.append("method_result_binding_missing")
    elif context.method_result_digest != canonical_digest(method.model_dump(mode="json")):
        failures.append("method_result_mismatch")
    request = context.method_request
    if request is None:
        unknowns.append("method_request_missing")
        return failures, unknowns
    provenance = method.provenance
    comparisons = (
        (request.case_id, evidence.case_id, "method_case_mismatch"),
        (request.ab_mapping_id, evidence.ab_mapping_id, "ab_mapping_mismatch"),
        (request.sample_digest, provenance.sample_digest, "method_sample_mismatch"),
        (request.config_digest, provenance.config_digest, "method_config_mismatch"),
        (request.method_version, provenance.method_version, "method_config_mismatch"),
        (request.configuration_version, provenance.configuration_version, "method_config_mismatch"),
        (request.threshold, method.threshold, "method_config_mismatch"),
        (request.selected_row_count, provenance.selected_row_count, "method_sampling_mismatch"),
        (request.training_row_count, provenance.training_row_count, "method_sampling_mismatch"),
        (request.evaluation_row_count, provenance.evaluation_row_count, "method_sampling_mismatch"),
        (request.sampling_seed, provenance.sampling_seed, "method_sampling_mismatch"),
        (request.split_seed, provenance.split_seed, "method_sampling_mismatch"),
    )
    failures.extend(code for expected, actual, code in comparisons if expected != actual)
    return failures, unknowns


def _numeric_claim_equal(actual, expected, *, bound):
    """Finite scalar/pair comparison; tolerance is enabled only for bound citations.

    Never use for hashes, identities, counts, indices, method settings or categories.
    Scalar/vector shape and A/B ordering remain exact. No payload is normalized.
    """
    if isinstance(actual, tuple) or isinstance(expected, tuple):
        return (
            isinstance(actual, tuple)
            and isinstance(expected, tuple)
            and len(actual) == len(expected)
            and all(
                _numeric_claim_equal(a, b, bound=bound)
                for a, b in zip(actual, expected, strict=True)
            )
        )
    if type(actual) not in (int, float) or type(expected) not in (int, float):
        return False
    if not math.isfinite(actual) or not math.isfinite(expected):
        return False
    return actual == expected or (
        bound
        and math.isclose(
            actual, expected, rel_tol=NUMERIC_CLAIM_REL_TOL, abs_tol=NUMERIC_CLAIM_ABS_TOL
        )
    )


def _check_claim(evidence, claim, *, numeric_bound=False):
    """Small typed vocabulary; no text interpretation or statistical estimation."""
    if isinstance(claim, MethodDirectionClaim):
        method = evidence.method_result
        if method is None:
            return "fail", "method_unavailable"
        if method.status == "error":
            return "fail", "method_execution_error"
        if method.threshold is None:
            return "fail", "method_threshold_unconfigured"
        if claim.direction != method.predicted_direction:
            return "fail", "method_direction_mismatch"
        return "pass", "method_claim_consistent"
    if isinstance(claim, DisplayRowClaim):
        if evidence.display_rows is None:
            return "fail", "rows_unavailable"
        if claim.row_index >= len(evidence.display_rows):
            return "fail", "row_unavailable"
        if not _numeric_claim_equal(
            claim.values, evidence.display_rows[claim.row_index], bound=numeric_bound
        ):
            return "fail", "row_value_mismatch"
        return "pass", "row_claim_consistent"
    if evidence.summary is None:
        return "fail", "summary_unavailable"
    expected = getattr(evidence.summary, claim.statistic)
    if expected is None:
        return "fail", "statistic_unavailable"
    # n_display is a discrete count, not a floating-point measurement.
    matches = (
        claim.value == expected
        if claim.statistic == "n_display"
        else _numeric_claim_equal(claim.value, expected, bound=numeric_bound)
    )
    if not matches:
        return "fail", "summary_value_mismatch"
    return "pass", "summary_claim_consistent"
