---
summary: 'Mapping from the legacy `pnpm mcp:*` scripts to the modern mcporter CLI.'
read_when:
  - 'Helping teammates migrate old pnpm workflows to mcporter'
---

# Migrating from `pnpm mcp:*`

The legacy `pnpm mcp:*` helpers map directly onto the `mcporter` CLI.

- `pnpm mcporter:list` → `npx mcporter list`
- `pnpm mcporter:call server.tool key=value` → `npx mcporter call server.tool key=value`
- New flags: `--schema` surfaces full tool schemas, and `--tail-log` follows log output referenced by responses.

For a step-by-step checklist (including config updates and environment variables) see [`docs/migration.md`](./migration.md).
