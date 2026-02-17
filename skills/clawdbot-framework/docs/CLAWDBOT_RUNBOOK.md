# Runbook
How to run the demo, the robustness suite, and the nonseparable fork test

This runbook consolidates operational instructions and expected artifacts.

## Install

1. Install dependencies
   pip install -r requirements.txt

2. Optional
   If using the web UI, ensure flask is installed (it is listed in README dependencies).

## Canonical capacity → geometry experiment

Run
   python scripts/run_capacity_dimshift.py

Common options
   python scripts/run_capacity_dimshift.py --preset small
   python scripts/run_capacity_dimshift.py --preset large
   python scripts/run_capacity_dimshift.py --D 4 --N 16

Artifacts
Outputs are written to outputs/capacity_dimshift/<run_id>/ and include:
1. metadata.json
2. sweep_results.csv
3. ds_matrix.json
4. thresholds.json
5. heatmap.png
6. representative_curves.png
7. phase_diagram.png

## Robustness suite

Run
   python scripts/run_robustness_suite.py

Quick subset suitable for CI
   python scripts/run_robustness_suite.py --quick

Artifacts
Outputs are written to outputs/capacity_dimshift_suite/<suite_id>/ with per run folders plus:
1. suite_summary.csv
2. suite_metadata.json

## Web interface

Run
   python app.py

Open
   http://localhost:5000

## Nonseparable Laplacian fork test (Step 1)

Run
   python scripts/run_nonseparable_rewire_test.py

If the script supports presets, prefer:
1. small: D=3, N=16, exact
2. medium: D=3, N=32, exact
3. large: D=3, N=64, SLQ

Artifacts
Outputs are written to outputs/nonseparable_rewire/<run_id>/ and include:
1. metadata.json
2. sweep_results.csv
3. thresholds.json
4. summary.json

Interpretation tips
1. Random long range edges can raise absolute d_s values due to shortcuts.
2. The acceptance criteria focus on staircase structure, monotonicity, and jump locations, not absolute plateau height.

## Test and validation commands

Unit tests
   pytest -q

Focused tests
   pytest -q tests/test_theorem_validation.py
   pytest -q tests/test_audit.py
   pytest -q tests/test_nonseparable_step1.py

Validation harness for theorem option B (if present)
   python scripts/run_framework_validation_b.py

