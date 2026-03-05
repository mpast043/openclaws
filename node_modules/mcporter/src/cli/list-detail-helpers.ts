import type { GeneratedOption } from './generate/tools.js';
import { buildDocComment } from './list-doc-comments.js';
import type { FlagUsageExtra } from './list-signature.js';
import {
  formatCallExpressionExample,
  formatExampleBlock,
  formatFlagLabel,
  formatFlagUsage,
  formatFunctionSignature,
  formatOptionalSummary,
} from './list-signature.js';
import { extraDimText } from './terminal.js';

export type { FlagUsageExtra } from './list-signature.js';
export {
  formatCallExpressionExample,
  formatExampleBlock,
  formatFlagLabel,
  formatFlagUsage,
  formatFunctionSignature,
  formatOptionalSummary,
} from './list-signature.js';

export interface SelectDisplayOptionsResult {
  displayOptions: GeneratedOption[];
  hiddenOptions: GeneratedOption[];
}

export interface ToolDocInput {
  serverName: string;
  toolName: string;
  description?: string;
  outputSchema?: unknown;
  options: GeneratedOption[];
  requiredOnly: boolean;
  colorize?: boolean;
  exampleMaxLength?: number;
  flagExtras?: FlagUsageExtra[];
  defaultReturnType?: string;
  callSelector?: string;
  wrapExampleExpression?: boolean;
}

export interface ToolDocModel {
  docLines?: string[];
  signature: string;
  tsSignature: string;
  flagUsage: string;
  optionalSummary?: string;
  examples: string[];
  displayOptions: GeneratedOption[];
  hiddenOptions: GeneratedOption[];
  optionDocs: ToolOptionDoc[];
}

export interface ToolOptionDoc {
  option: GeneratedOption;
  flagLabel: string;
  description: string;
}

const DEFAULT_MIN_VISIBLE_PARAMS = 5;
export function selectDisplayOptions(
  options: GeneratedOption[],
  requiredOnly: boolean,
  minVisible = DEFAULT_MIN_VISIBLE_PARAMS
): SelectDisplayOptionsResult {
  if (!requiredOnly || options.length <= minVisible) {
    return { displayOptions: options, hiddenOptions: [] };
  }
  const included = new Set<number>();
  options.forEach((option, index) => {
    if (option.required) {
      included.add(index);
    }
  });
  let includedCount = included.size;
  if (includedCount < minVisible) {
    for (let index = 0; index < options.length && includedCount < minVisible; index += 1) {
      if (included.has(index)) {
        continue;
      }
      included.add(index);
      includedCount += 1;
    }
  }
  const displayOptions = options.filter((_option, index) => included.has(index));
  const hiddenOptions = options.filter((_option, index) => !included.has(index));
  return { displayOptions, hiddenOptions };
}

export function buildToolDoc(input: ToolDocInput): ToolDocModel {
  const {
    serverName,
    toolName,
    description,
    outputSchema,
    options,
    requiredOnly,
    colorize = true,
    exampleMaxLength,
    flagExtras,
    defaultReturnType,
  } = input;
  const { displayOptions, hiddenOptions } = selectDisplayOptions(options, requiredOnly);
  const docLines = buildDocComment(description, options, { colorize });
  const signature = formatFunctionSignature(toolName, displayOptions, outputSchema, {
    colorize,
    defaultReturnType,
  });
  const tsSignature = formatFunctionSignature(toolName, displayOptions, outputSchema, {
    colorize: false,
    defaultReturnType,
  });
  const flagUsage = formatFlagUsage(displayOptions, flagExtras, { colorize });
  const optionalSummary = hiddenOptions.length > 0 ? formatOptionalSummary(hiddenOptions, { colorize }) : undefined;
  const optionDocs = options.map((option) => buildOptionDoc(option, { colorize }));
  const callExample = formatCallExpressionExample(
    serverName,
    toolName,
    displayOptions.length > 0 ? displayOptions : options,
    { callSelector: input.callSelector, wrapExpression: input.wrapExampleExpression }
  );
  const examples = callExample
    ? formatExampleBlock([callExample], { maxExamples: 1, maxLength: exampleMaxLength ?? 80 })
    : [];
  return {
    docLines,
    signature,
    tsSignature,
    flagUsage,
    optionalSummary,
    examples,
    displayOptions,
    hiddenOptions,
    optionDocs,
  };
}

function buildOptionDoc(option: GeneratedOption, opts?: { colorize?: boolean }): ToolOptionDoc {
  return {
    option,
    flagLabel: formatFlagLabel(option),
    description: formatOptionDescription(option, opts),
  };
}

function formatOptionDescription(option: GeneratedOption, opts?: { colorize?: boolean }): string {
  const colorize = opts?.colorize ?? false;
  const tint = colorize ? extraDimText : (value: string): string => value;
  let description = option.description ? option.description : `Set ${option.property}.`;
  const detailParts: string[] = [];
  if (option.enumValues && option.enumValues.length > 0) {
    detailParts.push(`choices: ${option.enumValues.join(', ')}`);
  }
  if (option.defaultValue !== undefined) {
    detailParts.push(`default: ${formatHelpValue(option.defaultValue)}`);
  }
  if (option.exampleValue) {
    detailParts.push(`example: ${option.exampleValue}`);
  }
  if (detailParts.length > 0) {
    description += ` (${detailParts.join('; ')})`;
  }
  return tint(description);
}

function formatHelpValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((entry) => String(entry)).join(', ');
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (value && typeof value === 'object') {
    return JSON.stringify(value);
  }
  return String(value);
}
