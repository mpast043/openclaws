#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

const rootDir = path.resolve(process.argv[2] ?? process.cwd());

const server = new McpServer({ name: 'fs-fixture', version: '1.0.0' });

server.registerTool(
  'list_files',
  {
    title: 'List Files',
    description: 'List the files in the configured root',
    inputSchema: {},
    outputSchema: {
      files: z.array(z.string()),
    },
  },
  async () => {
    const entries = await fs.readdir(rootDir);
    return {
      content: [{ type: 'text', text: entries.join('\n') }],
      structuredContent: { files: entries },
    };
  }
);

server.registerTool(
  'read_text_file',
  {
    title: 'Read Text File',
    description: 'Read a UTF-8 file relative to the MCP root',
    inputSchema: {
      path: z.string().describe('Relative path inside the root directory'),
    },
    outputSchema: {
      contents: z.string(),
    },
  },
  async ({ path: relativePath }) => {
    const targetPath = path.resolve(rootDir, relativePath);
    if (!targetPath.startsWith(rootDir)) {
      throw new Error('path escapes configured root');
    }
    const data = await fs.readFile(targetPath, 'utf8');
    return {
      content: [{ type: 'text', text: data }],
      structuredContent: { contents: data },
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
await new Promise((resolve, reject) => {
  transport.onclose = resolve;
  transport.onerror = reject;
});
