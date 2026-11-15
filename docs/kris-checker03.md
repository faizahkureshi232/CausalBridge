# Checker 0.3 numeric evidence fix and deterministic replay

Base branch: `codex/kris-track-b-plan` at
`1638d674af7a411ef4cc9d7b38fd6db5bc15927e`, plus uncommitted infrastructure
changes from the first/second smoke work. This change adds only the bound numeric
claim policy, checker version, contract and regression controls. No prompts,
model settings, evidence packages or cached predictions are changed.

## Scope

`verifier/checks.py` now uses math.isclose(rel_tol=1e-12, abs_tol=1e-12) for
finite continuous display-row and summary claim values, after strict citation
validation and exact evidence/proposal/receipt binding. Pair shapes and A/B
positions remain exact. No comparison searches other fields for a close value.
IDs, digests, row indices, counts (including n_display), seeds, method settings,
categories, method directions and reason codes retain their exact semantics.
No numeric data or digest input is rounded or normalized by this change.

NaN/infinity and malformed values fail closed through existing schema/canonical
JSON checks. Null/missing statistics and invalid references fail before tolerant
comparison. The helper also checks finiteness explicitly. Invalid nonfinite
objects bypassing constructors can raise ValueError at the public boundary;
callers must record the failure and must never release it.

PredictionCheck's default is now `direction-checker-0.3`. This changes its
version-bound replay cache key; original checker-0.2 decisions remain intact.
The integration contract documents rationale, formula, scope and limitations.

## Saved smoke-v2 comparison

| Package | Original checker 0.2 | Updated checker 0.3 | Updated release |
| --- | --- | --- | --- |
| E1 | pass, prediction_abstained | pass, prediction_abstained | None |
| E2 | pass, prediction_abstained | pass, prediction_abstained | None |
| E3 | pass, prediction_abstained | pass, prediction_abstained | None |
| E4 | fail, summary_value_mismatch | pass, evidence_consistent | A_to_B |

E4/S1's copied mean differs by 8.326672684688674e-17. Both method-direction
claims remain correct and all three claim findings now pass. The separately
observed E4/S0 row difference, 1.3877787807814457e-17, is also covered by the
regression suite. S0 is not added to the experimental S2 matrix.

The first smoke is a negative control: E1 remains rejected for rows_unavailable;
E2/E3 remain rejected for method_unavailable; E4 remains passed/released A_to_B.
No unintended decision changes were observed in these saved S1 proposals.

## Replay provenance and independent audit

Copied both original smoke bundles to separately named checker03-replay
folders, preserving generations, original invocation exports, requests and
method inputs/results byte-for-byte. Called existing IntegratedRunner.replay_s2
only, twice per manifest to verify checker-version cache reuse. Patched model
generation, predict_direction and execute_anm_request to raise if called;
all call counts were zero. The independent native auditor passed both new
bundles with zero findings. Old journal release events remain present alongside
the new checker-0.3 events; active S2 exports reference the updated checks.

Replay artifacts:

- `runs/qwen3-8b-live-smoke-02-v2-checker03-replay/replay-report.json`
- `runs/qwen3-8b-live-smoke-02-v2-checker03-replay/replay-audit.json`
- `runs/qwen3-8b-live-smoke-02-v2-checker03-replay/analysis-checker03.jsonl`
- Equivalent first-smoke controls under
  `runs/qwen3-8b-live-smoke-01-checker03-replay/`.

The copied root report.json/audit.json and original invocation exports are
historical snapshots. replay-report.json/replay-audit.json and the versioned
analysis file describe the new replay. Original source bundles remain unchanged:
37 files in smoke 1 and 38 in smoke 2 have identical before/after SHA-256 hashes.
Each replay-report records those hashes and the checker source hash.

## Validation and freeze assessment

54 new parameterized regression cases cover both observed mismatches, exact
matches, zero/negative/scalar/pair values, immediate floating-point neighbors
inside/outside absolute and relative boundaries, material differences, wrong
citation/shape/row/statistic, receipt/proposal identities and hashes, exact
method provenance/settings, nonfinite/missing values, and categorical directions.
Tests prove math.isclose is not called for unbound numeric citations. Historical
fixture assertions compare semantic decisions separately from checker version;
fixture files and their original check versions are not rewritten.

Full `scripts/check.sh`: **566 tests passed**, Ruff lint/format passed, both
Track A demos, real synthetic ANM handshake, wheel and source distribution passed.
Both saved-smoke replay bundles pass independent native audit. No live calls,
commits, pushes or merges occurred.

The numeric policy is technically ready to freeze as checker 0.3 within its
explicit scope. It does not establish causal correctness, validate unrestricted
reason prose, require nonempty citations or require agreement with ANM. Absolute
tolerance can accept tiny nonzero values near zero; relative tolerance permits
larger absolute offsets at huge magnitudes. These are documented policy effects,
not proof of model understanding or statistical reliability. If the team wants a
minimum-support rule or a different numeric policy, decide/version it before the
checker freeze and before held-out evaluation. Overall real-data/scorer readiness
is a separate gate.
