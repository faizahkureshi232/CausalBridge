# First live smoke review and separately versioned second smoke

Base branch: `codex/kris-track-b-plan`, commit
`1638d674af7a411ef4cc9d7b38fd6db5bc15927e`, plus local uncommitted changes.
The original `runs/qwen3-8b-live-smoke-01` bundle is read-only historical evidence.
Do not rerun the old v1 entry point into that directory after changing source code:
source changes invalidate its native cache identity. Use the v2 command below.

## Findings and fixes

1. E2/S0 and E3/S0 each began **12 display-row claims**, ending mid-JSON at
   `eval_count=1024`, `done_reason=length`. Their first claimed row values do not
   match the actual cited rows either. The original schema permits an unbounded
   claims tuple and the prompt supplies no length/citation-count instruction.
   The observed mechanism is verbose, partly invented row enumeration; the model's
   internal reason for choosing it cannot be established from outputs alone.
2. `OllamaModel` previously raised before constructing a Generation on length
   termination. Therefore neither the durable transport nor PredictionExecution
   received raw text/usage. It now builds Generation with `status=truncated`,
   `failure_reason=output_token_limit`, raw HTTP body, text, model, tokens and
   latency; GenerationFailure carries that record. The runner persists it before
   rethrowing, the predictor keeps it with `failure_stage=generate` and no parsed
   prediction/context, and crash recovery also refuses to parse it. Even valid
   JSON terminated by length remains non-scorable. No-response failures retain
   generation=null and unknown usage; token counts are never fabricated.
3. The first prompt/context budgets did not reserve the full output cap. Exact
   saved Ollama prompt counts are below. The 40,960-token model architecture limit
   does not establish DSMLP GPU allocation capacity. This environment cannot reach
   the Mac tunnel, has no DSMLP GPU telemetry/host connection, and cannot certify
   8192 support. The v2 entry point reads `/api/ps` without generation; selecting
   8192 requires the pinned model already loaded on GPU with context_length>=8192.
   It fails before scheduling trials if that evidence is absent.
4. Empty claims are permitted. The existing checker validates identities, original
   receipts, method bindings and each supplied typed claim, then releases a passing
   directional answer; no minimum claim count or ANM-agreement rule exists.
   E4/S1's zero-claim pass establishes no checked supporting citation. This rule is
   unchanged. Exports now distinguish binding validity, prediction scorability,
   supplied/checked claim counts, no-claims status, released direction and unknown
   causal correctness pending separate evaluator labels. A process pass is not
   a causal correctness label.
5. Native audit limitations previously hardcoded "scripted model transport" even
   for the live bundle. Audit v2 reports model_modes separately from
   observation_kinds. New bundles declare `scripted_fixture` or `live_ollama`
   independently of `synthetic` or `real_observational`. Legacy missing provenance
   remains `unspecified`; old reports are not rewritten. Provenance declarations
   are not an independent dataset eligibility certification.

## Versions and code boundaries

- Protocol: `integrated-track-b-v2-compact`; v1 remains selectable and unchanged.
- Prompt: `direction-0.2-compact`; historical `direction-0.1` remains unchanged.
- Pilot config: `qwen3-smoke-v2-compact-display16-ctx4096-out512`.
- Export: `native-analysis-v2`; audit: `native-handshake-audit-v2`.
- `evaluation/integration.py`: optional protocol/provenance configuration, shared
  concise instructions in S0/S1, durable failed generations and safe recovery.
- `llm/client.py`: additive Generation failure metadata and typed failure carrier.
- `llm/predictor.py`: preserve failed response, skip parsing, record prompt version.
- `evaluation/export.py`: explicit binding/claim/release/scoring-scope reporting.
- `evaluation/audit_native.py`: independent v1/v2 prompt expectations, HTTP/raw
  identity/status binding, non-scorable truncation and usage accounting checks,
  truthful transport/observation metadata.
- `evaluation/integration_demo.py`: optional v2 protocol and synthetic declaration.
- `examples/live_track_b_smoke_v2.py`: new entry point, same IntegratedRunner;
  original entry point unchanged. No replacement runner, ANM or verifier.
- `tests/test_smoke_v2.py`: thirteen regression cases, including rehashed semantic
  corruptions so audit rejection does not rely on stale file/journal hashes.

The new shared instruction requests concise JSON, a short reason, at most three
relevant claims, no row enumeration, no invented evidence, and only available
rows/statistics/method evidence. Abstention is still allowed with null direction,
abstained=true and a reason. S1 retains exactly its additional caution instruction.
DirectionPrediction and its output JSON schema are unchanged. The claim limit is
an instruction, not a silently added checker release criterion; violations remain
possible and require inspection.

## Context measurements and conservative alternative

| Package | v1 S0 input | v1 S1 input | v1 S1 + 1024 output cap |
| --- | ---: | ---: | ---: |
| E1 | 393 | 432 | 1456 |
| E2 | 3207 | 3246 | 4270 |
| E3 | 3216 | 3255 | 4279 |
| E4 | 3792 | 3831 | 4855 |

Default v2 uses **16 displayed rows**, **4096 context**, **512 output tokens**.
The complete original 128-row method sample, orientation, source digest, seeds,
ANM config and uncalibrated smoke threshold are retained. The original seeded
selector chooses the new display rows; E2/E3/E4 still share identical display
rows and recomputed display-only summaries. E1 retains its names-only visibility.
These are separate pilot conditions, not a prompt-only comparison against v1.

A per-request character-ratio estimate calibrated to the saved v1 token counts
puts worst-case v2 input at about **2874 tokens** (E4/S1). Input + 512 generation
+ 512 reserve is **3898 < 4096**. This is an estimate, not exact Qwen tokenization
or proof against server truncation. `runs/kris-smoke-v2-validation/context-plan.json`
contains all eight exact old counts, new message sizes and estimates. Inspect
actual v2 prompt_eval_count, termination reasons and server context telemetry after
running; no live generation was used to estimate the budget.

An optional 8192 configuration is supported by the entry point only after its
metadata preflight confirms an already loaded GPU model at that capacity. It
retains the 512-token output cap. It neither loads/restarts the model nor changes
DSMLP allocations to obtain that evidence. The conservative default avoids making
an unsupported claim about remote VRAM headroom.

## Validation

Full `scripts/check.sh` passes: **512 tests**, Ruff lint/format, both Track A demos,
actual synthetic ANM handshake, wheel and source distribution. The thirteen new
cases cover valid-JSON truncation, metadata/tokens, no-response distinction,
interruption recovery, zero-call cache reuse, release semantics/reporting,
transport/data provenance, v2 wire settings and 8192 preflight rejection,
and independent audit rejection of altered usage/truncation status.

Both v1 and v2 native integration demos pass their independent audits: 24 scripted
calls and 12 saved S1 checks each. Repeating v2 issues zero calls. The historical
live bundle also passes a fresh read-only audit. All original live bundle file
hashes remain unchanged. Validation artifacts are under
`runs/kris-smoke-v2-validation`; they are synthetic/scripted test evidence.

## Run the second live smoke from the Mac terminal

Keep the existing SSH tunnel open. From a separate Mac terminal:

```bash
cd "/Users/krisdcosta/Documents/ChatGPT/DSC 261 Responsible Data Science/CausalBridge"
PYTHONPATH=src .venv/bin/python examples/live_track_b_smoke_v2.py \
  --baseline runs/live-track-b-smoke-baseline-20261114 \
  --output runs/qwen3-8b-live-smoke-02-v2
```

This pins `qwen3:8b` Q4_K_M digest
`500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`,
temperature=0, seed=261, think=false, context=4096, output=512, display=16,
method sample=128. Session identity includes effective settings, adapter hash,
commit/branch and all Python source hashes, including uncommitted modifications.
Changed sources/configuration require a new output directory. The script issues
exactly eight primary generation requests for a fresh run, no automatic retries,
four deterministic S2 replay records (or explicit parent-failure skips), native
audit and a same-session zero-call cache verification. `/api/tags` and `/api/ps`
are metadata reads, not additional generations. Repeating the same command with
unchanged sources/settings reuses all terminal outcomes, including failures.

Send `runs/qwen3-8b-live-smoke-02-v2/report.json` for inspection. The report includes
all eight outcomes plus the v2 analysis export and checker results. Original first
smoke files are never used as an output destination.

Remaining risks: instructions may still be ignored; 512 tokens may still be
insufficient for a verbose answer; exact v2 token sizes and GPU 8192 capacity are
unmeasured; empty-claim releases do not validate support; threshold 0.001 is
uncalibrated; fixed E1-to-E4/S0-then-S1 order is retained and cannot establish
order-independent effects. No real Tuebingen data, scorer handoff or statistical
accuracy claim is established. No live second run, commits, pushes or merges are
performed during this infrastructure change.

## Phase 1 reconciliation after PR #7

The earlier validation/run instructions above describe the original preparation
snapshot. A second live smoke has since been saved in the documented output
directory; its report records eight live calls, no extra S2/resume calls and a
passing original audit. Do not overwrite it. Checker 0.3 replay is separate.

The branch is now based on merged PR #7 (`3d9d645`), which adds the Track A
ten-group pilot without changing Track B interfaces. Phase 1 reviewed the
compact prompts, durable failure metadata, provenance exports and bound numeric
claim tolerance. Focused tests passed **117**; full repository checks passed
**569 tests**, lint/format, both demos, synthetic ANM handshake and package builds.
The current source changes remain uncommitted. No new live calls were made in
this phase. See the integration contract for failure and numeric policies.

## PR #8 review follow-up

The native auditor now independently derives completion status and failure reason
from every saved HTTP response: length termination is truncated; done other than
literal true is incomplete; missing/empty textual content is incomplete. Stored
status/reason must match, and incomplete raw responses cannot carry a scorable
prediction or verification context even when their stored status claims complete.

The analysis export counts claims whenever a parsed prediction exists, including
binding failures. This does not make the prediction valid, scorable, checked or
releasable. Parsed empty claims report zero; absent parsed predictions report null.

18 additional cases cover clean and forged/rehashed HTTP completion states,
synchronized exports/S2 receipts, binding-failure claim retention, empty claims,
and generation/parse failures. Validation: 587 tests passed, lint/format, both
Track A demos, both native protocol demos and independent audits, ANM handshake,
and wheel/sdist builds. Native repeats made zero new scripted calls. A fresh
frozen environment sync was unavailable because the pinned Ruff wheel was absent
from the offline cache; validation used the existing installed environment for
all test/check commands. No new live generations, prompt changes, checker-policy
changes or historical artifact edits were required.
