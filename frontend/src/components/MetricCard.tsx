import React from 'react';
import { formatMetric } from '../lib/format';

interface MetricCardProps {
  label: string;
  value: number | null | undefined;
  isPercentage?: boolean;
  description?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  isPercentage = true,
  description,
}) => {
  const isEvaluated = value !== null && value !== undefined && !isNaN(value);

  return (
    <div className="bg-surface rounded-[12px] border border-border p-4 shadow-xs flex flex-col justify-between transition-all hover:border-slate-300">
      <div>
        <span className="text-[11px] font-semibold text-navy-muted uppercase tracking-wider block mb-0.5">
          {label}
        </span>
        {description && (
          <p className="text-[11px] text-navy-muted mb-2 leading-tight">
            {description}
          </p>
        )}
      </div>

      <div className="mt-2">
        {isEvaluated ? (
          <span className="text-xl sm:text-2xl font-black text-navy-foreground tracking-tight font-mono tabular-nums">
            {formatMetric(value, isPercentage)}
          </span>
        ) : (
          <span className="text-xs font-medium text-navy-muted italic bg-canvas px-2 py-0.5 rounded-[6px] border border-border/60">
            Not evaluated
          </span>
        )}
      </div>
    </div>
  );
};
