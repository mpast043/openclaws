# TOOLS.md

Local notes for this workspace (environment specific).

Repository and runtime
1. Python: use a virtual environment (.venv) and install requirements from the repo.
2. Tests: run pytest before committing changes.
3. Common scripts (adjust paths to match your repo):
   1. Capacity sweep demo: python scripts/run_capacity_dimshift.py
   2. Nonseparable test run: python scripts/run_nonseparable_rewire_test.py
4. When a result matters, capture:
   1. command line used
   2. git commit hash
   3. config parameters
   4. output artifact paths

OpenClaw tool suite (recommended for this agent)
1. Keep tools.profile at full or (coding + add web/ui/automation/messaging).
2. Use allow/deny lists to match your comfort level.
3. For host execution, prefer allowlist + ask in Exec Approvals.

See docs/OPENCLAW_SETUP_GUIDE.md for a concrete config snippet and safety defaults.
