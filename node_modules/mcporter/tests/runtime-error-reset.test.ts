import { ErrorCode, McpError } from '@modelcontextprotocol/sdk/types.js';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createRuntime } from '../src/runtime.js';

describe('runtime connection resets', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('closes cached connections after fatal MCP errors', async () => {
    const runtime = await createRuntime({ servers: [] });
    type ClientContext = Awaited<ReturnType<typeof runtime.connect>>;
    const rejected = new McpError(ErrorCode.ConnectionClosed, 'Connection closed');
    const context = {
      client: {
        callTool: vi.fn().mockRejectedValue(rejected),
      },
      transport: { close: vi.fn().mockResolvedValue(undefined) },
      definition: {
        name: 'temp',
        description: 'test',
        command: { kind: 'stdio', command: 'node', args: [], cwd: process.cwd() },
        source: { kind: 'local', path: '<test>' },
      },
      oauthSession: undefined,
    } as unknown as ClientContext;
    vi.spyOn(runtime, 'connect').mockResolvedValue(context);
    (runtime as unknown as { clients: Map<string, Promise<ClientContext>> }).clients.set(
      'temp',
      Promise.resolve(context)
    );
    const closeSpy = vi.spyOn(runtime, 'close').mockResolvedValue();

    await expect(runtime.callTool('temp', 'list_pages')).rejects.toThrow('Connection closed');
    expect(closeSpy).toHaveBeenCalledWith('temp');
  });

  it('keeps the connection open for user-facing InvalidParams errors', async () => {
    const runtime = await createRuntime({ servers: [] });
    type ClientContext = Awaited<ReturnType<typeof runtime.connect>>;
    const rejected = new McpError(ErrorCode.InvalidParams, 'Tool help not found');
    const context = {
      client: {
        callTool: vi.fn().mockRejectedValue(rejected),
      },
      transport: { close: vi.fn().mockResolvedValue(undefined) },
      definition: {
        name: 'temp',
        description: 'test',
        command: { kind: 'stdio', command: 'node', args: [], cwd: process.cwd() },
        source: { kind: 'local', path: '<test>' },
      },
      oauthSession: undefined,
    } as unknown as ClientContext;
    vi.spyOn(runtime, 'connect').mockResolvedValue(context);
    (runtime as unknown as { clients: Map<string, Promise<ClientContext>> }).clients.set(
      'temp',
      Promise.resolve(context)
    );
    const closeSpy = vi.spyOn(runtime, 'close').mockResolvedValue();

    await expect(runtime.callTool('temp', 'help')).rejects.toThrow('Tool help not found');
    expect(closeSpy).not.toHaveBeenCalled();
  });
});
