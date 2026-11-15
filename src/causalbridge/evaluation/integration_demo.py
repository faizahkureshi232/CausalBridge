"""Actual ANM/predictor/checker on synthetic pairs and scripted model replies.

No real-world evaluation, LLM inference or threshold tuning is performed.
"""

import argparse
import json
import random
from pathlib import Path

from causalbridge.discovery.anm_requests import ANMConfig
from causalbridge.discovery.contracts import RuntimePair
from causalbridge.discovery.evidence import select_pair
from causalbridge.evaluation.export import analysis_rows, evaluator_mapping
from causalbridge.evaluation.integration import (
    COMPACT_PROTOCOL,
    PROTOCOL,
    IntegratedRunner,
    build_native_packages,
)
from causalbridge.llm.client import Generation


class ScriptedNativeModel:
    def generate(self, messages, schema):
        user = json.loads(messages[1]["content"])
        evidence = user["evidence"]
        method = evidence["method_result"]
        direction = None if evidence["package"] == "E1" else "A_to_B"
        claims = []
        if method and method["status"] == "ok" and method["threshold"] is not None:
            direction = method["predicted_direction"]
            claims = [
                dict(type="method_direction", evidence_ref="method_result", direction=direction)
            ]
        reply = dict(
            case_id=user["case_id"],
            package_digest=user["package_digest"],
            predicted_direction=direction,
            abstained=direction is None,
            reason_code="scripted_smoke_abstention" if direction is None else None,
            claims=claims,
        )
        return Generation(text=json.dumps(reply), model="scripted-native-integration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=(PROTOCOL, COMPACT_PROTOCOL), default=PROTOCOL)
    args = parser.parse_args()
    packages = []
    for i in range(3):
        rng = random.Random(73 + i)
        rows = []
        for _ in range(128):
            x = rng.uniform(-2, 2)
            rows.append((1.0, 2.0) if i == 1 else (x, x * x + rng.gauss(0, 0.2)))
        pair = RuntimePair(
            case_id=f"D{i + 1:04d}",
            variables=({"name": "Measurement one"}, {"name": "Measurement two"}),
            rows=rows,
        )
        selected = select_pair(pair)
        config = ANMConfig(
            sampling_seed=selected.selection.spec.sample_seed,
            threshold=None if i == 2 else 0.001,
            configuration_version="integration-fixture-only",
            threshold_selection_version="not-calibrated-smoke-test",
        )
        packages.extend(build_native_packages(selected, args.output / "anm" / pair.case_id, config))
    runner = IntegratedRunner(
        args.output / "experiment",
        model_identity="scripted-native-v1",
        identity_verified=True,
        model_mode="scripted_fixture",
        observation_kind="synthetic",
        protocol=args.protocol,
    )
    run = runner.run(packages, ScriptedNativeModel())
    releases = runner.replay_s2(run["manifest_id"])
    export = analysis_rows(run, releases)
    (args.output / "evaluation_mapping.json").write_text(
        json.dumps(evaluator_mapping(packages), indent=2) + "\n"
    )
    (args.output / "analysis.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in export)
    )
    print(
        json.dumps(
            dict(
                notice="Real computation/components, synthetic data/scripted LLM replies",
                manifest_id=run["manifest_id"],
                new_model_calls=run["new_model_calls"],
                cached_or_recovered_trials=run["cached_or_recovered_trials"],
                s2_records=len(releases),
                analysis_rows=len(export),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
