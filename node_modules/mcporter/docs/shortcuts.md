---
summary: 'Agent-friendly shortcuts for getting help from configured MCP servers.'
read_when:
  - 'You need a quick reminder of which hidden CLI shortcuts explain MCP servers.'
---

# Agent Shortcuts

These undocumented shortcuts are safe for MCP agents to call when they need a quick description of a server or its tools without memorizing the full `mcporter` CLI surface.

## `pnpm mcp call <server>.help`

- Alias for `pnpm mcporter:call <server>.help`, meant for agents that already rely on the old `pnpm mcp:*` helpers.
- If `<server>` exposes a `help` tool, the call behaves exactly like any other tool invocation and streams the tool's content.
- When no `help` tool exists, the CLI automatically falls back to `mcporter list <server>` so the agent still sees the server summary, tool signatures, and copy-pasteable examples.
- Use `--output json` if the agent prefers machine-readable summaries; the fallback will mirror that flag.

## `mcporter describe <server>`

- Hidden synonym for `mcporter list <server>`; it prints the same schema-rich output without teaching agents about the `list` verb.
- Works with any selector the `list` command accepts (configured names, ad-hoc `https://` URLs, or `--stdio/--http-url` flags).
- Pair with `--schema` or `--all-parameters` when an agent needs the full JSON Schema for every tool.

### Recommended Usage

1. Agents wanting prose guidance should run `pnpm mcp call chrome-devtools.help`.
2. If the server lacks a `help` tool, the command emits a dim hint and then shows the `describe`/`list` output so the agent still learns about the server.
3. For a guaranteed TypeScript-style summary, skip straight to `mcporter describe chrome-devtools --schema`.
4. Need the tool menu immediately? Call `pnpm mcp call chrome-devtools.list_tools`—it’s a shortcut for `mcporter list chrome-devtools`.

> Note: `chrome-devtools` currently ships without a `help` tool, so step 1 always triggers the fallback and prints the same schema-rich output you would see from `mcporter list`.
