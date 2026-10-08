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
    <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs flex flex-col justify-between">
      <div>
        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-1">
          {label}
        </span>
        {description && (
          <p className="text-[11px] text-slate-400 mb-2 leading-tight">
            {description}
          </p>
        )}
      </div>

      <div className="mt-2">
        {isEvaluated ? (
          <span className="text-2xl font-black text-slate-900 tracking-tight font-mono">
            {formatMetric(value, isPercentage)}
          </span>
        ) : (
          <span className="text-sm font-semibold text-slate-400 italic">
            Not evaluated
          </span>
        )}
      </div>
    </div>
  );
};
