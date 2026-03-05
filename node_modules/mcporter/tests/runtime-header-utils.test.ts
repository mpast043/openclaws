import { afterEach, describe, expect, it } from 'vitest';
import { materializeHeaders } from '../src/runtime-header-utils.js';

describe('materializeHeaders', () => {
  const envKey = 'HEADER_TOKEN';
  let previousValue: string | undefined;

  afterEach(() => {
    if (previousValue === undefined) {
      delete process.env[envKey];
    } else {
      process.env[envKey] = previousValue;
    }
    previousValue = undefined;
  });

  it('resolves environment placeholders in headers', () => {
    previousValue = process.env[envKey];
    process.env[envKey] = 'test-secret';
    const headers = materializeHeaders({ Authorization: `Bearer \${HEADER_TOKEN}` }, 'inline');
    expect(headers).toEqual({ Authorization: 'Bearer test-secret' });
  });

  it('propagates a helpful error when placeholder resolution fails', () => {
    previousValue = process.env[envKey];
    delete process.env[envKey];
    expect(() => materializeHeaders({ Authorization: `Bearer \${HEADER_TOKEN}` }, 'broken')).toThrow(
      /Failed to resolve header 'Authorization'/
    );
  });

  it('returns undefined when no headers are provided', () => {
    const headers = materializeHeaders(undefined, 'noop');
    expect(headers).toBeUndefined();
  });
});
