import type { ServerDefinition } from '../config.js';
import { normalizeHttpUrl } from './http-utils.js';

export function findServerByHttpUrl(definitions: readonly ServerDefinition[], urlString: string): string | undefined {
  const normalizedTarget = normalizeHttpUrl(urlString);
  if (!normalizedTarget) {
    return undefined;
  }
  for (const definition of definitions) {
    if (definition.command.kind !== 'http') {
      continue;
    }
    const normalizedDefinitionUrl = normalizeHttpUrl(definition.command.url);
    if (!normalizedDefinitionUrl) {
      continue;
    }
    if (normalizedDefinitionUrl === normalizedTarget) {
      return definition.name;
    }
  }
  return undefined;
}
