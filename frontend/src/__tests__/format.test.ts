import { describe, it, expect } from 'vitest';
import { formatProbability, formatMetric, formatScore } from '../lib/format';

describe('Format Utilities', () => {
  it('formatProbability returns "Not evaluated" for null and undefined', () => {
    expect(formatProbability(null)).toBe('Not evaluated');
    expect(formatProbability(undefined)).toBe('Not evaluated');
    expect(formatProbability(NaN)).toBe('Not evaluated');
  });

  it('formatProbability formats numbers to one decimal percentage without altering values', () => {
    expect(formatProbability(0.87)).toBe('87.0%');
    expect(formatProbability(0.9125)).toBe('91.3%');
    expect(formatProbability(0.0)).toBe('0.0%');
  });

  it('formatMetric returns "Not evaluated" for null and formats numbers correctly', () => {
    expect(formatMetric(null)).toBe('Not evaluated');
    expect(formatMetric(undefined)).toBe('Not evaluated');
    expect(formatMetric(0.892, true)).toBe('89.2%');
    expect(formatMetric(0.892, false)).toBe('0.892');
  });

  it('formatScore returns "Not evaluated" or 3 decimal places', () => {
    expect(formatScore(null)).toBe('Not evaluated');
    expect(formatScore(0.884)).toBe('0.884');
  });
});
