import { describe, expect, it, vi } from 'vitest';
import { handleList as runHandleList } from '../src/cli/list-command.js';
import type { ServerDefinition } from '../src/config.js';
import type { Runtime } from '../src/runtime.js';

const healthyDefinition: ServerDefinition = {
  name: 'healthy',
  command: { kind: 'http', url: new URL('https://healthy.example.com/mcp') },
};

const authDefinition: ServerDefinition = {
  name: 'auth-server',
  command: { kind: 'http', url: new URL('https://auth.example.com/mcp') },
};

function createRuntime(): Runtime {
  const definitions = [healthyDefinition, authDefinition];
  return {
    getDefinitions: () => definitions,
    getDefinition: (name: string) => {
      const definition = definitions.find((entry) => entry.name === name);
      if (!definition) {
        throw new Error(`Unknown server '${name}'`);
      }
      return definition;
    },
    registerDefinition: vi.fn(),
    listTools: vi.fn((name: string) => {
      if (name === 'healthy') {
        return Promise.resolve([{ name: 'list_documents' }]);
      }
      return Promise.reject(new Error('HTTP error 401: auth required'));
    }),
  } as unknown as Runtime;
}

describe('handleList JSON output', () => {
  it('emits aggregated status counts', async () => {
    const runtime = createRuntime();
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    await runHandleList(runtime, ['--json']);

    const payload = JSON.parse(logSpy.mock.calls.at(-1)?.[0] ?? '{}');
    expect(payload.mode).toBe('list');
    expect(payload.counts.auth).toBe(1);
    const healthyEntry = payload.servers.find((entry: { name: string }) => entry.name === 'healthy');
    expect(healthyEntry.status).toBe('ok');
    const authEntry = payload.servers.find((entry: { name: string }) => entry.name === 'auth-server');
    expect(authEntry.status).toBe('auth');
    expect(authEntry.issue.kind).toBe('auth');

    logSpy.mockRestore();
  });
});
