---
name: clawdbot-safety
description: Safety checklist for enabling tools, exec approvals, and installing third-party ClawHub skills
user-invocable: true
disable-model-invocation: true
---

# clawdbot-safety

When invoked, produce a short checklist that:
1. Confirms current tool policy (tools.profile, allow/deny).
2. Confirms exec approvals settings (deny/allowlist/full, ask mode).
3. Inspects any third-party skill before enabling (read SKILL.md, look for risky prerequisites).
4. Recommends running risky commands only in sandbox where possible.
