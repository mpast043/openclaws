# OPENCLAW_SETUP_GUIDE.md

This is a practical setup guide for enabling a "full suite" of OpenClaw tools and skills for clawdbot, while staying safe.

1. Tool access policy
OpenClaw tool access is controlled by tools.profile plus allow/deny lists. Tool groups let you enable capabilities in bundles (fs, runtime, web, ui, automation, messaging, memory, nodes). See the OpenClaw Tools docs.

Suggested starting point (capable but still bounded)
Use a coding base, then add the main operational groups you actually want:

```json
{
  "tools": {
    "profile": "coding",
    "allow": ["group:web", "group:ui", "group:automation", "group:messaging"],
    "deny": []
  },
  "skills": {
    "load": { "watch": true, "watchDebounceMs": 250 }
  }
}
```

If you want everything available (least restrictive), set tools.profile to full (or omit it) and remove allow/deny. Consider pairing that with Exec Approvals in allowlist mode.

2. Exec safety (strongly recommended)
Host execution uses exec approvals as a safety interlock. Keep security at allowlist and ask at on-miss or always.

3. Web search
If you want web_search/web_fetch, you will need the configured provider and keys (for example BRAVE_API_KEY for Brave search).

4. Skills
Workspace skills live in <workspace>/skills and override bundled skills by name. Keep your custom skills in skills/ and restart a session to pick them up.

5. Third-party skill safety
Treat third-party skills as untrusted. Inspect SKILL.md and any scripts before enabling.

Files in this workspace:
1. docs/OBJECTIVES.md: put your explicit objectives here (this drives “what good looks like”).
2. docs/CAPABILITY_MAP.md: maps common asks to tools and safe patterns.
