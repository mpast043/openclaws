import fsPromises from 'node:fs/promises';
import { describe, expect, it, vi } from 'vitest';

import { MCPORTER_VERSION } from '../src/runtime.js';

process.env.MCPORTER_DISABLE_AUTORUN = '1';
const cliModulePromise = import('../src/cli.js');

describe('mcporter --version', () => {
  it('reads package.json when available, falling back to runtime version on failure', async () => {
    const { runCli } = await cliModulePromise;
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const readSpy = vi.spyOn(fsPromises, 'readFile').mockRejectedValueOnce(new Error('missing'));

    await runCli(['--version']);

    expect(logSpy).toHaveBeenCalledWith(MCPORTER_VERSION);
    expect(readSpy).toHaveBeenCalled();

    logSpy.mockRestore();
    readSpy.mockRestore();
    process.exitCode = undefined;
  });
});
