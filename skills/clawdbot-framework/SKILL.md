---
name: clawdbot-framework
description: Framework v4.5–v4.6 Capacity→Geometry operations, theorem obligations, and Step 1 nonseparable runbook
user-invocable: false
metadata: {"openclaw":{"requires":{"bins":["python3","git"],"config":["tools.exec"]}}}
---

# clawdbot-framework

This skill teaches the agent how to operate inside Megan’s Framework v4.5–v4.6 Capacity→Geometry workspace.

Operating invariants
1. Capacity-only: vary capacity; hold substrate, estimator, windows, and seeds fixed unless a redesign is explicit.
2. Respect the tier split: algebraic theorems, asymptotic facts, empirical obligations.
3. Run tests before reporting results when feasible, and record provenance (params, commit hash, artifact paths).

Primary references (local)
1. {baseDir}/docs/CLAWDBOT_CONTEXT.md
2. {baseDir}/docs/CLAWDBOT_RUNBOOK.md
3. {baseDir}/docs/CLAWDBOT_ACCEPTANCE.md
4. {baseDir}/docs/THEOREMS.md
5. {baseDir}/docs/FRAMEWORK_THEOREM_B.md
6. {baseDir}/docs/EXPERIMENT_NONSEPARABLE_REWIRE.md

When asked to do work
1. Start by reading the relevant reference doc above.
2. If code changes are requested, propose the smallest change that preserves invariants and test coverage.
3. If results are requested, run the script/tests and provide exact commands and outputs.
