# WORKFLOW_AUTO.md

Framework with Selection end-to-end
MCP-first, autonomy-enabled, evidence-retaining workflow
Authoritative runbook: this file only

## 0. Intent

This workflow is designed to run end-to-end without handholding:

1. Validate the host-adapters repo and CGF layer.
2. Execute baseline scientific tests.
3. Autonomously design and run additional high-value tests (new ideas) to resolve underdetermined claims.
4. Run Selection across the full “Framework with Selection” proposal using the produced evidence.
5. Record, retain, and summarize everything with reproducible artifacts.

Key constraint: minimize laptop workload.
Default behavior: use MCP tools for compute, document parsing, retrieval, and long sweeps whenever available.
Local execution is reserved for lightweight repo checks and orchestration.

## 1. Autonomy policy

The workflow must not ask “STOP vs CONTINUE” for routine issues.
It must classify failures and proceed according to the rules below.

Autonomy is enabled in two phases:
A. Baseline phase (required): run the standard suite.
B. Exploration phase (required): generate and run additional tests the agent proposes, bounded by time and budget.

The agent is allowed to propose and execute new tests if all are true:

1. The test produces measurable evidence tied to a claim or falsifier.
2. The test is reproducible (seeded where applicable) and writes artifacts.
3. The test does not require semantic refactors, new physics claims, or new major dependencies.

If a proposed test requires any of the following, it must stop and ask for operator approval:

1. Changing schemas or public API shapes.
2. Adding new external dependencies beyond what is already in requirements.
3. Large code refactors (more than mechanical formatting).
4. Claims that materially expand the scope of the Framework proposal beyond the current document.

## 1.1 Runtime anti-stall policy

This workflow must avoid long single-turn blocking runs.

1. If a command is expected to run for more than 90 seconds, launch it as a non-blocking process job and poll every 20 to 30 seconds.
2. Do not keep a single agent turn open longer than 8 minutes; emit current status and checkpoint paths, then continue in the next cycle.
3. Always continue with `--resume-latest` after timeout, disconnect, or policy interruption.
4. Write a cycle checkpoint on every supervisor iteration to `results/supervisor/cycle_status_latest.json`.
5. Use bounded per-cycle budgets by default:
   `--cycle-timeout-seconds 480 --cycle-local-only-max-minutes 8 --cycle-local-only-max-runs 6`.

## 2. MCP-first compute policy

2.1 Default compute placement
Prefer MCP for:

1. Parameter sweeps and long campaigns
2. Model selection bootstraps and resampling
3. PDF parsing and chunking
4. Vector search / retrieval indexing for Selection
5. Any run expected to take more than 2 minutes or more than 1 CPU core

Local is allowed for:

1. make test
2. contract suite if it is quick
3. starting CGF server (unless MCP offers a server runtime)
4. short “smoke” scientific runs only if MCP compute is unavailable

2.2 MCP discovery and selection
At the start of the workflow:

1. Use mcporter to list MCP servers and tools.
2. Select one “compute-capable” MCP target (preferred order):
   a. remote runner with job execution
   b. container runtime
   c. hosted python/jupyter execution
3. Select one “document/retrieval” MCP target if present (PDF parsing, embedding, vector DB, search).
4. Record the chosen MCP targets and tool names in manifest.json.

If no suitable MCP targets exist, proceed locally but mark verdict as LOCAL_ONLY and reduce scope (no long campaigns).

2.3 Workload minimization rules

1. Never run a long sweep locally if any MCP compute is available.
2. If MCP supports asynchronous jobs, submit jobs and poll results with bounded polling (do not busy wait).
3. Pull back only the minimal artifact set needed to validate conclusions:
   manifest, raw data, fits/model selection outputs, verdicts, and summary.

## 3. Required response format

For each step, the agent response must include the following sections:

1. Step
   Step X name, start time, end time, PASS/FAIL/PARTIAL, and artifact paths.

2. What I decided
   One paragraph describing:
   a. what was attempted
   b. what changed
   c. why (linking to this workflow’s rules)

3. Commands and MCP calls
   Copy-pasteable command list and MCP tool calls (names and key arguments).
   Do not include secrets.

4. Outputs written
   List paths for logs and results written this step.

5. Next step
   One line: the next step identifier.

## 4. Artifact contract, logging, and retention

4.1 RUN_DIR
Create RUN_DIR at repo root:
RUN_YYYYMMDD_HHMMSS

Inside RUN_DIR:
logs/
results/
tmp/
manifest.json
summary.md

4.2 Logging contract
For every local command:

1. Capture stdout/stderr to RUN_DIR/logs/<name>.txt
2. If exit code nonzero, also write tail200 to RUN_DIR/logs/<name>_tail200.txt
3. Record command, cwd, exit code, start/end timestamps into manifest.json

For every MCP tool call:

1. Write an entry to manifest.json with:
   tool name, server target, request summary, response pointer
2. Store returned job ids and artifact locations
3. If MCP returns logs, store them under RUN_DIR/logs/mcp_<name>.txt

4.3 Results index
Write RUN_DIR/results/index.json containing pointers to:
contracts/
sdk_validation/
science/
selection/
VERDICT.json

4.4 Retention
Retention is mandatory.

1. Create RUN_DIR/retained_RUN_YYYYMMDD_HHMMSS.tar.gz
2. Copy it to repo_root/retained_runs/
3. Write RUN_DIR/RETAINED.txt with:
   archive filename, size, retention path, timestamp
4. If MCP provides storage upload, also upload the archive and record remote URI in RETAINED.txt

## 5. Failure handling (self-troubleshooting)

5.1 Classifications
Classify each failure as:
LINT_FAILURE
TEST_FAILURE
CONTRACT_FAILURE
RUNTIME_FAILURE
SCIENCE_EVIDENCE_FAILURE
SELECTION_FAILURE

5.2 Default decisions
A. LINT_FAILURE

1. Attempt safe autofix once (formatters/linters only).
2. Re-run lint.
3. If remaining lint is noncritical (formatting, unused imports, naming):
   continue to tests and record LINT_DEBT.
4. If critical (syntax/parsing, undefined names, import/module failures):
   stop.

B. TEST_FAILURE
Stop. Report failing tests, traceback tail, rerun command.

C. CONTRACT_FAILURE
Stop. Report failing contract test, rerun command, suspected surface.

D. RUNTIME_FAILURE
Attempt up to two safe fixes, then stop.

E. SCIENCE_EVIDENCE_FAILURE
Do not treat as workflow failure.
Record as a scientific verdict outcome (REJECTED or NOT SUPPORTED), retain artifacts, and continue to Selection with the recorded outcome.

F. SELECTION_FAILURE
Stop. Selection outputs are required for completion.

5.3 Safe fixes allowed without approval

1. ruff/black/isort formatting and autofix only
2. changing ports for local server conflicts
3. setting documented default environment variables
4. restarting a local process that failed to start

Anything else requires approval.

## 6. Inputs

Required:

1. Repository with Makefile, sdk/, adapters/, server/, tools/, experiments/
2. policy/policy_config_v03.json
3. Framework with Selection proposal PDF available locally or retrievable via MCP

PDF discovery order:

1. repo root: "Framework with selection.pdf" or "Framework with Selection.pdf"
2. docs/ or spec/ directories
3. if missing, retrieve via MCP (document store, drive connector, or a provided path)

If the PDF cannot be found or retrieved, stop as SELECTION_FAILURE.

## 7. Workflow steps start to finish

Step 0 Preflight and MCP discovery
0.1 Locate repository root and record git state
0.2 Create RUN_DIR structure
0.3 Record environment:
os, python, pip freeze, git remote/branch/commit, dirty status
0.4 MCP discovery:

* command -v mcporter
* mcporter list
* for each candidate server, inspect available tools (mcporter inspect if supported)
  0.5 Choose MCP targets:
* compute_target
* docs_target (optional)
  Record both in manifest.json.

Stop if mcporter is missing (RUNTIME_FAILURE).

Step 1 Repo verification (minimal local workload)
1.1 Install deps (local)
pip install -r requirements.txt
1.2 Lint (local, with autofix policy)
make lint
If ruff is available, allow:
ruff check . --fix
ruff format .
1.3 Tests (local)
make test
Hard stop on test failure.

Step 2 CGF layer bring-up and contract verification (prefer MCP if supported)
2.1 Start CGF server
Preferred:

* if compute_target can run a server job, run CGF server under MCP
  Fallback:
* start locally

Default endpoint: http://127.0.0.1:8080
If port conflict, select: 18080, 28080, 38080.
Record chosen endpoint in manifest.json.

2.2 Health check
Use the simplest available method:

* SDK ping if exists
* HTTP GET health endpoint if exists
  Store response to RUN_DIR/results/cgf_health.json

2.3 Contract suite
Preferred command:
bash tools/run_contract_suite.sh
Fallback:
python tools/contract_compliance_tests.py
Store under RUN_DIR/results/contracts and logs.

Hard stop on contract failure.

2.4 SDK artifact validation
python tools/validate_sdk_artifacts.py
Store under RUN_DIR/results/sdk_validation

Hard stop on failure.

Step 3 Scientific baseline suite (MCP compute preferred)
The goal is to create high-quality evidence artifacts first, then expand.

3.1 Baseline run plan
Create RUN_DIR/results/science/run_plan.json with:

* baseline_tests (required)
* exploration_tests (to be generated later)
* budgets and stop conditions

3.2 Locate baseline runners
Identify canonical scripts in experiments/:

* Claim 2 runner (tradeoff)
* Claim 3 v4 entropy scaling + falsifiers
* Claim 3P physical convergence runner

If a baseline runner cannot be located, stop as RUNTIME_FAILURE.

3.3 Execute baseline tests
Run baseline tests on compute_target if possible.
For each run, write:
RUN_DIR/results/science/<name>/<run_id>/
with manifest/metadata, raw data, fits/model selection outputs, verdict json.

Baseline results must be summarized into:
RUN_DIR/results/science/baseline_summary.json

SCIENCE_EVIDENCE_FAILURE is allowed and recorded as REJECTED or NOT SUPPORTED.
Do not stop just because a claim is rejected.

Step 4 Exploration phase (agent proposes and runs new tests)
This is where the agent “comes up with its own ideas and tests.”

4.1 Build a candidate test set
Generate 5 to 12 tests that maximize information gain for the Framework proposal.
Each proposed test must include:

* target claim(s)
* hypothesis
* measurable metric(s)
* acceptance/falsifier condition
* estimated cost (time/cores)
* placement (MCP compute or local)
  Write proposals to:
  RUN_DIR/results/science/exploration_proposals.json

4.2 Select an execution subset
Choose a subset that fits within budget and has highest expected value.
Default budget:

* max_minutes: 120 total exploration time (MCP time)
* max_runs: 80
* max_failures: 10
* seeds: at least 3 distinct seeds where applicable

Write selected tests to:
RUN_DIR/results/science/exploration_selected.json

4.3 Execute selected tests
Run on compute_target whenever possible.
Maintain:
RUN_DIR/results/science/campaign/campaign_index.csv
Append one row per run immediately:
timestamp, test_id, model, L, chi, seed, key_metrics, aic_delta, bic_delta, verdict, artifact_path

At campaign end, write:
RUN_DIR/results/science/campaign/campaign_report.md
RUN_DIR/results/science/campaign/model_comparison.json

4.4 Autonomy guardrails for exploration
Allowed exploration examples:

* expand chi beyond 16 to improve model identifiability
* repeat across seeds to stabilize AIC/BIC comparisons
* compare alternative plausible model families
* perform ablations (remove a constraint, change noise clipping) with explicit labeling
* robustness checks (CV, bootstrap CI)
* scaling checks with L

Not allowed without approval:

* new frameworks, new schemas, major refactors, new dependencies

Step 5 Selection pass on the Framework with Selection proposal (MCP docs/retrieval preferred)
Goal: produce a claim ledger with witnesses that ties the proposal to current evidence.

5.1 Build evidence index
Create RUN_DIR/results/selection/evidence_index.json mapping:

* each claim id -> relevant artifacts (paths) and short descriptors
  Include:
  contracts, sdk validation, baseline science, exploration science.

5.2 Run Selection
Preferred:

* use docs_target to parse/chunk the PDF and build retrieval index
* use compute_target to execute the selection/verifier pipeline
  Fallback:
* run locally only if lightweight and no MCP option exists

Selection outputs are required:
RUN_DIR/results/selection/selection_manifest.json
RUN_DIR/results/selection/ledger.jsonl
RUN_DIR/results/selection/selection_report.md

Ledger requirements:

* status: ACCEPTED, UNDERDETERMINED, REJECTED
* rationale
* witness pointers to local artifact paths and, when possible, span identifiers
* at least one underdetermined item includes a next-test recommendation

Hard stop if Selection outputs are missing (SELECTION_FAILURE).

Step 6 Final verdict, summary, and retention
6.1 Write VERDICT.json
RUN_DIR/results/VERDICT.json must include:

* overall_status: COMPLETE, PARTIAL, STOPPED
* lint_status: PASS, LINT_DEBT, FAIL_CRITICAL
* contract_status
* scientific_status: per test suite + key metrics
* selection_status
* deltas vs baseline expectations
* key artifact pointers

6.2 Write summary.md
RUN_DIR/summary.md must include:

1. What ran (baseline + exploration)
2. What was learned and what changed
3. Current conclusions (with evidence pointers)
4. What is still underdetermined and the next best tests
5. Troubleshooting and fixes applied
6. Exact rerun commands and MCP tool call recipes

6.3 Retain artifacts
Create archive and copy to retained_runs plus optional MCP storage upload.
Write RETAINED.txt.

6.4 Cleanup
If the workflow started a CGF server locally, stop it and record cleanup in logs.

## 8. Troubleshooting quick playbook

Port conflict for CGF

* rotate ports 8080 -> 18080 -> 28080 -> 38080, record selection

Lint failure explosion

* save logs and tail200
* use make -n lint to identify tool
* if ruff available: ruff check . --fix then ruff format .
* treat noncritical as debt and proceed to tests

Missing MCP capabilities

* proceed locally but shrink exploration:
  max_minutes 20, max_runs 10, no large sweeps
* record LOCAL_ONLY in VERDICT.json

Model selection instability

* expand chi grid and add seeds
* bootstrap AIC/BIC differences
* report both per-run and pooled model comparison

Selection lacks witnesses

* rebuild evidence_index.json with explicit pointers
* ensure selection pipeline uses artifact paths as primary ground truth
* rerun Selection

End of WORKFLOW_AUTO.md
