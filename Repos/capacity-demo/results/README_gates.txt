Drop files into your run output directory and run:

python v45_apply_step2_step3.py --root /path/to/run_outputs

Expected (minimum) files:
- metrics.json (or run_metrics.json): numeric metrics for Step 2 gates
- selection.jsonl: selection records for Step 3 gates

The script writes gates_report.json and gates_report.md into the same root.
