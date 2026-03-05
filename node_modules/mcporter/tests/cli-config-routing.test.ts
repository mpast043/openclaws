import { describe, expect, it, vi } from 'vitest';

process.env.MCPORTER_DISABLE_AUTORUN = '1';

vi.mock('../src/cli/config-command.js', async () => {
  const actual = await vi.importActual<typeof import('../src/cli/config-command.js')>('../src/cli/config-command.js');
  return {
    ...actual,
    handleConfigCli: vi.fn().mockResolvedValue(undefined),
  };
});

describe('mcporter config entrypoint', () => {
  it('routes to the config handler before runtime inference', async () => {
    const { runCli } = await import('../src/cli.js');
    const { handleConfigCli } = await import('../src/cli/config-command.js');
    await runCli(['config']);
    expect(handleConfigCli).toHaveBeenCalledTimes(1);
  });
});
