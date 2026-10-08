export function formatProbability(value: number | null | undefined): string {
  if (value === null || value === undefined || isNaN(value)) {
    return 'Not evaluated';
  }
  return `${(value * 100).toFixed(1)}%`;
}

export function formatMetric(
  value: number | null | undefined,
  isPercentage = true
): string {
  if (value === null || value === undefined || isNaN(value)) {
    return 'Not evaluated';
  }
  if (isPercentage) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return value.toFixed(3);
}

export function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined || isNaN(value)) {
    return 'Not evaluated';
  }
  return value.toFixed(3);
}

export function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return 'Not evaluated';
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return String(dateStr);
    return d.toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return String(dateStr);
  }
}
