"""PR #8 review regressions: independent HTTP completion and failed-binding claims."""

import json

import httpx
import pytest
from test_native_integration_runner import NativeModel, configuration, selected_pair
from test_smoke_v2 import runner

from causalbridge.evaluation.audit import _hash
from causalbridge.evaluation.audit_native import audit_native
from causalbridge.evaluation.export import analysis_rows
from causalbridge.evaluation.integration import build_native_packages
from causalbridge.llm.client import ModelConfig, OllamaModel


@pytest.fixture
def packages(tmp_path):
    return build_native_packages(selected_pair(), tmp_path / "anm", configuration())


def http_model_response(request, **overrides):
    data = json.loads(request.content)
    generation = NativeModel().generate(data["messages"], data["format"])
    return httpx.Response(
        200,
        json=dict(
            model="qwen3:8b",
            done=True,
            done_reason="stop",
            prompt_eval_count=20,
            eval_count=10,
            message={"content": generation.text},
        )
        | overrides,
    )


@pytest.mark.parametrize(
    "overrides,status,reason",
    [
        ({}, "complete", None),
        ({"done": False, "done_reason": "stop"}, "incomplete", "response_not_done"),
        ({"done": False, "done_reason": "other"}, "incomplete", "response_not_done"),
        ({"done": None}, "incomplete", "response_not_done"),
        ({"done": 1}, "incomplete", "response_not_done"),
        ({"done_reason": "length"}, "truncated", "output_token_limit"),
        ({"message": {"content": "  "}}, "incomplete", "missing_text_content"),
        ({"message": {}}, "incomplete", "missing_text_content"),
    ],
)
def test_audit_accepts_correctly_recorded_http_states(
    packages, tmp_path, overrides, status, reason
):
    config = ModelConfig()
    r = runner(tmp_path / "run", config)
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: http_model_response(request, **overrides))
    ) as client:
        run = r.run(packages, OllamaModel(config, client=client))
    r.replay_s2(run["manifest_id"])
    for trial in run["trials"]:
        trace = trial["outcome"]["execution"]
        assert trace["generation"]["status"] == status
        assert trace["generation"]["failure_reason"] == reason
        assert (trace["failure_stage"] is None) == (status == "complete")
        if status != "complete":
            assert trace["prediction"] is None and trial["outcome"]["context"] is None
    audit = audit_native(tmp_path / "run")
    assert audit["passed"], audit


@pytest.mark.parametrize(
    "raw_mutation",
    [
        {"done": False, "done_reason": "stop"},
        {"done": False, "done_reason": "other"},
        {"done": None},
        {"done": 1},
        {"message": {"content": "  "}},
        {"message": {}},
    ],
)
def test_audit_rejects_rehashed_incomplete_http_marked_scorable(packages, tmp_path, raw_mutation):
    # Start with coherent successful predictions, contexts and S2 releases. Alter only
    # raw completion evidence; keep the forged completed status and parsed predictions.
    root = tmp_path / "run"
    config = ModelConfig()
    r = runner(root, config)
    with httpx.Client(transport=httpx.MockTransport(http_model_response)) as client:
        run = r.run(packages, OllamaModel(config, client=client))
    r.replay_s2(run["manifest_id"])
    path = root / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
        data = event["data"]
        generation = (
            data.get("generation")
            if event["kind"] == "native_generated"
            else (data["execution"]["generation"] if event["kind"] == "native_finished" else None)
        )
        if generation is not None:
            body = json.loads(generation["raw_response"]) | raw_mutation
            generation["raw_response"] = json.dumps(body)
            if "message" in raw_mutation:
                generation["text"] = raw_mutation["message"].get("content", "")
    starts = {e["data"]["key"]: e["data"] for e in events if e["kind"] == "native_started"}
    ends = {e["data"]["key"]: e["data"] for e in events if e["kind"] == "native_finished"}
    # Rebind parent outcome digests and release keys so failures cannot be explained
    # by stale S2 parent receipts or journal hashes.
    for event in events:
        if event["kind"] == "native_released":
            release = event["data"]
            release["parent_outcome_digest"] = native_hash(ends[release["source_key"]])
            release["key"] = native_hash(
                [
                    release["source_key"],
                    ends[release["source_key"]],
                    release["check"]["checker_version"],
                ]
            )
    previous = None
    for event in events:
        event["previous"] = previous
        event["digest"] = _hash({k: event[k] for k in ("seq", "previous", "kind", "data")})
        previous = event["digest"]
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    export_path = root / "exports" / (run["manifest_id"] + ".json")
    export = json.loads(export_path.read_text())
    export["trials"] = [
        dict(start=starts[t["start"]["key"]], outcome=ends[t["start"]["key"]])
        for t in export["trials"]
    ]
    export_path.write_text(json.dumps(export))
    releases = [e["data"] for e in events if e["kind"] == "native_released"]
    (root / "exports" / (run["manifest_id"] + "-s2.json")).write_text(json.dumps(releases))
    report = audit_native(root)
    assert not report["passed"]
    codes = {f["code"] for f in report["findings"]}
    assert "http_completion_status" in codes
    assert "incomplete_generation_not_scorable" in codes
    assert not {"journal_integrity", "trial_export", "s2_parent", "s2_export"} & codes


def native_hash(value):
    import hashlib

    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "behavior,expected_count",
    [
        ("misbound", 1),
        ("malformed", None),
        ("timeout", None),
    ],
)
def test_export_retains_parsed_claims_without_certifying_binding(
    packages, tmp_path, behavior, expected_count
):
    r = runner(tmp_path / "run", mode="scripted_fixture")
    run = r.run(packages, NativeModel(behavior))
    releases = r.replay_s2(run["manifest_id"])
    rows = analysis_rows(run, releases)
    for row in rows:
        if row["package"] != "E4" or row["condition"] not in {"S0", "S1", "S2"}:
            continue
        assert row["supplied_claim_count"] == expected_count
        assert row["prediction_binding"] == "invalid_or_unavailable"
        assert row["prediction_scorable"] is False
        assert row["checked_claim_count"] == 0
        assert row["supporting_claims_status"] == "not_checked"
        assert row["predicted_direction"] is None and row["released_direction"] is None


def test_binding_failure_with_parsed_empty_claims_reports_zero(packages, tmp_path):
    r = runner(tmp_path / "run", mode="scripted_fixture")
    run = r.run(packages, NativeModel("misbound"))
    rows = analysis_rows(run, r.replay_s2(run["manifest_id"]))
    assert all(
        row["supplied_claim_count"] == 0
        for row in rows
        if row["package"] == "E1" and row["condition"] in {"S0", "S1", "S2"}
    )
