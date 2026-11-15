"""Mac-terminal live smoke entry point; delegates all trials/cache/S2 to IntegratedRunner.

Uses the unchanged integration demo's D0001 case and uncalibrated smoke ANM config.
No real-data findings, automatic retries, model downloads, or protocol changes.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from uuid import uuid4

import httpx

from causalbridge.discovery.anm_requests import load_anm_request
from causalbridge.evaluation.audit_native import audit_native
from causalbridge.evaluation.export import analysis_rows, evaluator_mapping
from causalbridge.evaluation.integration import (
    IntegratedRunner,
    NativePackage,
    build_native_packages,
)
from causalbridge.evaluation.store import atomic_json
from causalbridge.llm.client import ModelConfig, OllamaModel

EXPECTED_DIGEST = "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"


def model_metadata(client, base_url):
    response = client.get(base_url + "/api/tags")
    response.raise_for_status()
    matches = [m for m in response.json()["models"] if m["name"] == "qwen3:8b"]
    if len(matches) != 1 or matches[0]["digest"] != EXPECTED_DIGEST:
        raise ValueError("qwen3:8b is missing or its digest differs from Kris's pinned model")
    return matches[0]


class NoGeneration:
    """Prove resume cannot reach a model transport."""

    def __init__(self, config):
        self.config = config

    def generate(self, messages, schema):
        raise AssertionError("Cache verification attempted an extra generation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    args = parser.parse_args()
    root = args.output.resolve()
    base_url = args.base_url.rstrip("/")
    baseline_audit = audit_native(args.baseline / "experiment")
    if not baseline_audit["passed"]:
        raise ValueError("Baseline native audit failed")
    # Read original saved native packages, preserving selection/mapping/provenance.
    starts = [
        json.loads(line)["data"]
        for line in (args.baseline / "experiment" / "events.jsonl").read_text().splitlines()
        if json.loads(line)["kind"] == "native_started"
    ]
    originals = {
        s["package"]["evidence"]["package"]: NativePackage.model_validate(s["package"])
        for s in starts
        if s["package"]["evidence"]["case_id"] == "D0001"
    }
    if set(originals) != {"E1", "E2", "E3", "E4"}:
        raise ValueError("Expected the unchanged demo's D0001 E1-E4 packages")
    config = ModelConfig(
        model="qwen3:8b",
        base_url=base_url,
        temperature=0,
        seed=261,
        context_tokens=4096,
        max_tokens=1024,
        timeout_seconds=300,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    adapter_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # An HTTP hook adds the Ollama wire option without editing the shared transport.
    # The complete effective settings and adapter digest are bound into model_identity.
    effective = dict(
        model_digest=EXPECTED_DIGEST,
        think=False,
        config=config.model_dump(mode="json"),
        adapter_sha256=adapter_digest,
        source_commit=commit,
        source_branch=branch,
    )
    wire_calls = []

    def request_hook(request):
        if request.url.path != "/api/chat":
            return
        body = json.loads(request.content)
        body["think"] = False
        request._content = json.dumps(body).encode()
        request.stream = httpx.ByteStream(request.content)
        request.headers["content-length"] = str(len(request.content))
        call_id = uuid4().hex
        atomic_json(root / "wire" / (call_id + "-request.json"), body)
        wire_calls.append(call_id)
        request.extensions["smoke_call_id"] = call_id
        print(f"Live generation {len(wire_calls)}/8", flush=True)
        if len(wire_calls) > 8:
            raise AssertionError("Live call budget exceeded")

    def response_hook(response):
        if response.request.url.path == "/api/chat":
            response.read()
            atomic_json(
                root / "wire" / (response.request.extensions["smoke_call_id"] + "-response.json"),
                dict(
                    status_code=response.status_code,
                    body=response.text,
                ),
            )

    # Preflight is metadata only, never a generation. Fail before scheduling trials.
    with httpx.Client(timeout=300, trust_env=False) as preflight:
        metadata = model_metadata(preflight, base_url)
    root.mkdir(parents=True, exist_ok=True)
    session_path = root / "session.json"
    if session_path.exists():
        session = json.loads(session_path.read_text())
        if session["effective_settings"] != effective:
            raise ValueError("Saved session configuration changed; use a new output directory")
    else:
        session = dict(session_id=uuid4().hex, effective_settings=effective, model=metadata)
        atomic_json(session_path, session)
    native = originals["E4"]
    method_config = load_anm_request(Path(native.method_directory)).config
    packages = build_native_packages(native.selected, root / "anm" / "D0001", method_config)
    method = packages[3].evidence.method_result
    if method.status != "ok" or method.threshold is None:
        raise ValueError("Expected valid configured ANM evidence for this smoke case")
    runner = IntegratedRunner(
        root / "experiment",
        config=config,
        identity_verified=True,
        source_version=commit,
        model_identity=json.dumps(session, sort_keys=True),
    )
    with httpx.Client(
        timeout=300,
        trust_env=False,
        event_hooks={
            "request": [request_hook],
            "response": [response_hook],
        },
    ) as client:
        run = runner.run(packages, OllamaModel(config, client=client))
    atomic_json(root / "first-invocation.json", run)
    calls_before_s2 = len(wire_calls)
    releases = runner.replay_s2(run["manifest_id"])
    assert len(wire_calls) == calls_before_s2
    repeated = runner.run(packages, NoGeneration(config))
    assert repeated["new_model_calls"] == 0
    assert len(releases) == 4 and len(run["trials"]) == 8
    assert len(wire_calls) == run["new_model_calls"]
    atomic_json(root / "cache-verification.json", repeated)
    atomic_json(root / "evaluation_mapping.json", evaluator_mapping(packages))
    (root / "analysis.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in analysis_rows(run, releases))
    )
    audit = audit_native(root / "experiment")
    atomic_json(root / "audit.json", audit)
    outcomes = []
    for trial in run["trials"]:
        start, end = trial["start"], trial["outcome"]
        trace = end["execution"] or {}
        outcomes.append(
            dict(
                package=start["package"]["evidence"]["package"],
                condition=start["condition"]["condition_id"],
                prediction=trace.get("prediction"),
                failure_stage=trace.get("failure_stage"),
                error=trace.get("error") or end.get("operational_error"),
                generation=trace.get("generation"),
                latency_seconds=trace.get("latency_seconds"),
            )
        )
    report = dict(
        notice="Synthetic observations, real ANM and live Qwen; no Tuebingen findings",
        session=session,
        manifest_id=run["manifest_id"],
        new_model_calls=run["new_model_calls"],
        wire_calls=len(wire_calls),
        s2_new_model_calls=0,
        repeated_new_model_calls=repeated["new_model_calls"],
        outcomes=outcomes,
        checker_results=releases,
        audit_passed=audit["passed"],
        audit_findings=audit["findings"],
    )
    atomic_json(root / "report.json", report)
    print(json.dumps(report, indent=2))
    if not audit["passed"]:
        raise SystemExit("Native audit failed; inspect audit.json")


if __name__ == "__main__":
    main()
