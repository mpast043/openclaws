import type { GeneratedOption } from './generate/tools.js';
import { cyanText, extraDimText, yellowText } from './terminal.js';

const DEFAULT_WRAP_WIDTH = 100;

export function buildDocComment(
  description: string | undefined,
  options: GeneratedOption[],
  opts?: { colorize?: boolean }
): string[] | undefined {
  const colorize = opts?.colorize !== false;
  const descriptionLines = description?.split(/\r?\n/) ?? [];
  const paramDocs = options.filter((option) => option.description);
  if (descriptionLines.every((line) => line.trim().length === 0) && paramDocs.length === 0) {
    return undefined;
  }
  const tint = colorize ? extraDimText : (value: string): string => value;
  const highlightParam = colorize ? (value: string): string => yellowText(value) : (value: string): string => value;
  const highlightName = colorize ? (value: string): string => cyanText(value) : (value: string): string => value;
  const lines: string[] = [];
  lines.push(tint('/**'));
  let hasDescription = false;
  for (const line of descriptionLines) {
    const trimmed = line.trimEnd();
    if (trimmed.trim().length > 0) {
      const wrapped = wrapCommentText(trimmed);
      for (const segment of wrapped) {
        lines.push(tint(` * ${segment}`));
      }
      hasDescription = true;
    }
  }
  if (hasDescription && paramDocs.length > 0) {
    lines.push(tint(' *'));
  }
  for (const option of paramDocs) {
    const optionLines = formatParamDoc(option, DEFAULT_WRAP_WIDTH, {
      colorize,
      highlightParam,
      highlightName,
    });
    lines.push(...optionLines);
  }
  lines.push(tint(' */'));
  return lines;
}

function formatParamDoc(
  option: GeneratedOption,
  wrapWidth: number,
  config: {
    colorize: boolean;
    highlightParam: (value: string) => string;
    highlightName: (value: string) => string;
  }
): string[] {
  const { colorize, highlightParam, highlightName } = config;
  const descriptionLines = option.description?.split(/\r?\n/) ?? [''];
  const optionalSuffix = option.required ? '' : '?';
  const plainLabel = `@param ${option.property}${optionalSuffix}`;
  const continuationPrefix = colorize
    ? extraDimText(` * ${' '.repeat(plainLabel.length + 1)}`)
    : ` * ${' '.repeat(plainLabel.length + 1)}`;
  const rendered: string[] = [];
  descriptionLines.forEach((entry, index) => {
    const suffix = entry.trimEnd();
    if (index === 0) {
      const lineParts = [
        colorize ? extraDimText(' * ') : ' * ',
        highlightParam('@param '),
        highlightName(`${option.property}${optionalSuffix}`),
      ];
      if (suffix.length > 0) {
        const wrapped = wrapCommentText(suffix, wrapWidth - plainLabel.length - 1);
        if (wrapped.length > 0) {
          lineParts.push(colorize ? extraDimText(` ${wrapped[0]}`) : ` ${wrapped[0]}`);
          rendered.push(lineParts.join(''));
          for (const continuation of wrapped.slice(1)) {
            rendered.push(`${continuationPrefix}${colorize ? extraDimText(continuation) : continuation}`);
          }
          return;
        }
      }
      rendered.push(lineParts.join(''));
      return;
    }
    if (suffix.length > 0) {
      const wrapped = wrapCommentText(suffix, wrapWidth - plainLabel.length - 1);
      if (wrapped.length === 0) {
        return;
      }
      const [first, ...rest] = wrapped;
      if (!first) {
        return;
      }
      rendered.push(`${continuationPrefix}${colorize ? extraDimText(first) : first}`);
      for (const segment of rest) {
        rendered.push(`${continuationPrefix}${colorize ? extraDimText(segment) : segment}`);
      }
    }
  });
  return rendered;
}

export function wrapCommentText(text: string, width = DEFAULT_WRAP_WIDTH): string[] {
  if (!text) {
    return [];
  }
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length === 0) {
    return [];
  }
  const lines: string[] = [];
  let current = words[0] ?? '';
  for (let index = 1; index < words.length; index += 1) {
    const word = words[index];
    if (!word) {
      continue;
    }
    if (`${current} ${word}`.length > width) {
      lines.push(current);
      current = word;
    } else {
      current += ` ${word}`;
    }
  }
  lines.push(current);
  return lines;
}
