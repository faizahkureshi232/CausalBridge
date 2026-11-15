import pytest

from causalbridge.evaluation.diagnosis import diagnose_pilot


def _row(case, mapping, package, condition, direction, outcome=None, reasons=()):
    if outcome is None:
        outcome = "answer" if direction is not None else "model_abstention"
    return dict(
        case_id=case,
        ab_mapping_id=mapping,
        package=package,
        condition=condition,
        predicted_direction=direction,
        outcome=outcome,
        prediction_scorable=True,
        reason_codes=list(reasons),
    )


def cohort():
    """Two families. Weights make weighted accuracy differ from the unweighted rate."""
    evaluator = [
        dict(
            case_id="D1001",
            original_pair_id=1,
            reference_direction="first_to_second",
            weight=1.0,
            source_group="light",
            source_digest="x",
        ),
        dict(
            case_id="D1002",
            original_pair_id=2,
            reference_direction="second_to_first",
            weight=3.0,
            source_group="heavy",
            source_digest="y",
        ),
    ]
    mappings = [
        dict(
            case_id="D1001",
            ab_mapping_id="m1",
            source_digest="x",
            original_to_runtime_column_order=[0, 1],
        ),
        dict(
            case_id="D1002",
            ab_mapping_id="m2",
            source_digest="y",
            original_to_runtime_column_order=[0, 1],
        ),
    ]
    # Runtime references: D1001 A_to_B, D1002 B_to_A.
    answers = {
        ("E1", "S0"): ("A_to_B", "B_to_A"),
        ("E1", "S1"): ("A_to_B", None),
        ("E1", "S2"): ("A_to_B", None),
        ("E2", "S0"): (None, None),
        ("E2", "S1"): (None, None),
        ("E2", "S2"): (None, None),
        ("E3", "S0"): ("A_to_B", "A_to_B"),
        ("E3", "S1"): ("A_to_B", None),
        ("E3", "S2"): (None, None),
        ("E4", "S0"): ("A_to_B", "A_to_B"),
        ("E4", "S1"): ("A_to_B", "A_to_B"),
        ("E4", "S2"): ("A_to_B", "A_to_B"),
    }
    rows = []
    for package in ("E1", "E2", "E3", "E4"):
        for condition in ("S0", "S1", "S2"):
            left, right = answers[(package, condition)]
            rows.append(
                _row(
                    "D1001",
                    "m1",
                    package,
                    condition,
                    left,
                    outcome="fail" if package == "E3" and condition == "S2" else None,
                    reasons=("unavailable_evidence",)
                    if package == "E3" and condition == "S2"
                    else (),
                )
            )
            rows.append(_row("D1002", "m2", package, condition, right))
        for case, mapping, direction in (
            ("D1001", "m1", "A_to_B"),
            ("D1002", "m2", "B_to_A" if package != "E4" else "A_to_B"),
        ):
            rows.append(_row(case, mapping, package, "SIMPLE_GATE", direction, outcome="gate_pass"))
    rows.append(_row("D1001", "m1", "E4", "ANM_ONLY", "A_to_B", outcome="ok"))
    rows.append(_row("D1002", "m2", "E4", "ANM_ONLY", "A_to_B", outcome="ok"))
    anm = {
        "D1001": {
            "status": "ok",
            "predicted_direction": "A_to_B",
            "score_ab": 0.1,
            "score_ba": 0.15,
            "margin": 0.05,
            "threshold": 0.001,
        },
        "D1002": {
            "status": "ok",
            "predicted_direction": "A_to_B",
            "score_ab": 0.2,
            "score_ba": 0.202,
            "margin": 0.002,
            "threshold": 0.001,
        },
    }
    return rows, evaluator, mappings, anm


def test_weighted_and_unweighted_e4_rates_differ():
    rows, evaluator, mappings, anm = cohort()
    report = diagnose_pilot(rows, evaluator, mappings, anm)
    e4 = next(
        row for row in report["conditions"] if row["package"] == "E4" and row["condition"] == "S1"
    )
    assert e4["coverage"] == 1
    assert e4["selective_accuracy"] == 0.25
    assert e4["unweighted_coverage"] == 1
    assert e4["unweighted_selective_accuracy"] == 0.5
    assert report["total_weight"] == 4


def test_checker_counts_a_removed_correct_answer_separately():
    rows, evaluator, mappings, anm = cohort()
    report = diagnose_pilot(rows, evaluator, mappings, anm)
    e3 = next(row for row in report["checker"] if row["package"] == "E3")
    assert e3["removed_correct_count"] == 1
    assert e3["removed_correct_weight"] == 1
    assert e3["removed_wrong_count"] == 0
    assert e3["rejections"][0]["s2_reason_codes"] == ["unavailable_evidence"]
    e4 = next(row for row in report["checker"] if row["package"] == "E4")
    assert e4["removed_correct_count"] == e4["removed_wrong_count"] == 0


def test_contrasts_and_anm_copying():
    rows, evaluator, mappings, anm = cohort()
    report = diagnose_pilot(rows, evaluator, mappings, anm)
    contrasts = {row["name"]: row for row in report["contrasts"]}
    assert contrasts["H1_S1_E3_minus_E2"]["correct_yield"] == 0.25
    assert contrasts["H3_E4_S2_minus_S1"]["correct_yield"] == 0
    assert contrasts["H2_S1_E4_minus_ANM"]["selective_accuracy"] == 0
    e4_s1 = next(
        row
        for row in report["anm_agreement"]
        if row["package"] == "E4" and row["condition"] == "S1"
    )
    assert e4_s1["agree_answer"] == 2
    assert e4_s1["disagree"] == 0
    groups = {row["source_group"]: row for row in report["source_groups"]}
    assert groups["light"]["e4_s1_correct"] == 1
    assert groups["heavy"]["e4_s1_wrong"] == 1


def test_threshold_curve_uses_saved_margins_without_replacing_the_pilot_cut():
    rows, evaluator, mappings, anm = cohort()
    curve = diagnose_pilot(rows, evaluator, mappings, anm)["threshold_curve"]
    assert curve["status"] == "saved_margins_only"
    by_tau = {row["threshold"]: row for row in curve["rows"]}
    assert by_tau[0.001]["pilot_threshold"] is True
    assert by_tau[0.001]["selective_accuracy"] == 0.25
    assert by_tau[0.0]["coverage"] == 1
    assert by_tau[0.01]["coverage"] == 0.25
    assert by_tau[0.01]["selective_accuracy"] == 1


def test_missing_margins_do_not_invent_a_curve():
    rows, evaluator, mappings, _anm = cohort()
    curve = diagnose_pilot(rows, evaluator, mappings)["threshold_curve"]
    assert curve["status"] == "margins_unavailable"
    assert curve["rows"] == []


def test_anm_file_direction_must_match_the_export():
    rows, evaluator, mappings, anm = cohort()
    anm["D1002"]["predicted_direction"] = "B_to_A"
    with pytest.raises(ValueError, match="disagrees"):
        diagnose_pilot(rows, evaluator, mappings, anm)
