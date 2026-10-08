import React from 'react';
import { CheckCircle2, XCircle, MinusCircle, AlertTriangle } from 'lucide-react';
import { QualityInfo, OODInfo, QualityCheck } from '../types/analysis';

interface QualityPanelProps {
  quality: QualityInfo;
  ood: OODInfo;
}

export const QualityPanel: React.FC<QualityPanelProps> = ({ quality, ood }) => {
  const renderRow = (name: string, check: QualityCheck | null | undefined) => {
    if (!check || check.value === null || check.value === undefined) {
      return (
        <tr className="border-b border-border/60 last:border-0 text-xs">
          <td className="py-2.5 font-medium text-navy-foreground">{name}</td>
          <td className="py-2.5 text-navy-muted">Not evaluated</td>
          <td className="py-2.5 text-navy-muted">&mdash;</td>
          <td className="py-2.5 text-right">
            <span className="inline-flex items-center gap-1 text-navy-muted text-[11px]">
              <MinusCircle className="w-3.5 h-3.5" />
              <span>N/A</span>
            </span>
          </td>
        </tr>
      );
    }

    return (
      <tr className="border-b border-border/60 last:border-0 text-xs">
        <td className="py-2.5 font-medium text-navy-foreground">{name}</td>
        <td className="py-2.5 font-mono tabular-nums text-navy-foreground font-semibold">
          {check.value.toFixed(1)}
        </td>
        <td className="py-2.5 font-mono tabular-nums text-navy-muted">
          {check.threshold !== null ? `≥ ${check.threshold.toFixed(1)}` : '&mdash;'}
        </td>
        <td className="py-2.5 text-right">
          {check.passed === true ? (
            <span className="inline-flex items-center gap-1 text-clinical-success font-semibold text-[11px]">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Pass</span>
            </span>
          ) : check.passed === false ? (
            <span className="inline-flex items-center gap-1 text-clinical-danger font-semibold text-[11px]">
              <XCircle className="w-3.5 h-3.5" />
              <span>Fail</span>
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-navy-muted text-[11px]">
              <MinusCircle className="w-3.5 h-3.5" />
              <span>&mdash;</span>
            </span>
          )}
        </td>
      </tr>
    );
  };

  const qualityBadge = {
    acceptable: { bg: 'bg-emerald-50 text-emerald-800 border-emerald-200', label: 'Acceptable Quality' },
    poor: { bg: 'bg-rose-50 text-rose-800 border-rose-200', label: 'Degraded / Poor Quality' },
    not_evaluated: { bg: 'bg-slate-100 text-slate-700 border-slate-200', label: 'Not Evaluated' },
  }[quality.status];

  const oodLabel = ood.evaluated
    ? ood.is_ood
      ? 'Out of Distribution'
      : 'In-Domain (Chest X-Ray)'
    : 'Not Evaluated';

  const oodColor = ood.evaluated
    ? ood.is_ood
      ? 'bg-rose-50 text-rose-800 border-rose-200'
      : 'bg-teal-50 text-teal-800 border-teal-200'
    : 'bg-slate-100 text-slate-700 border-slate-200';

  return (
    <div className="bg-surface rounded-[12px] border border-border p-5 shadow-xs space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h3 className="text-sm font-bold text-navy-foreground tracking-tight">
            Image Quality & Domain Guardrails
          </h3>
          <span className="text-[11px] text-navy-muted">
            Heuristic clarity validation and Mahalanobis distance OOD filter
          </span>
        </div>
        <div className="flex gap-2">
          <span className={`text-[11px] px-2.5 py-1 rounded-full border font-semibold ${qualityBadge.bg}`}>
            {qualityBadge.label}
          </span>
          <span className={`text-[11px] px-2.5 py-1 rounded-full border font-semibold ${oodColor}`}>
            {oodLabel}
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-border text-[11px] uppercase tracking-wider text-navy-muted">
              <th className="pb-2 font-semibold">Quality Test</th>
              <th className="pb-2 font-semibold">Value</th>
              <th className="pb-2 font-semibold">Threshold</th>
              <th className="pb-2 font-semibold text-right">Status</th>
            </tr>
          </thead>
          <tbody>
            {renderRow('Laplacian Sharpness', quality.blur)}
            {renderRow('Mean Brightness', quality.brightness)}
            {renderRow('Contrast Variance', quality.contrast)}
          </tbody>
        </table>
      </div>

      {quality.reasons.length > 0 && (
        <div className="p-3 bg-rose-50 border border-rose-200 rounded-[10px] text-xs text-rose-900">
          <div className="flex items-center gap-1.5 font-bold mb-1 text-clinical-danger">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Image Quality Defect Detected</span>
          </div>
          <ul className="list-disc list-inside space-y-0.5 text-[11px]">
            {quality.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      {ood.is_ood && (
        <div className="p-3 bg-purple-50 border border-purple-200 rounded-[10px] text-xs text-purple-900">
          <span className="font-bold block mb-0.5">Out-of-Distribution Warning</span>
          <span className="text-[11px] leading-relaxed">
            {ood.reason || 'Input characteristics deviate from expected chest X-ray distribution. Inference halted.'}
          </span>
        </div>
      )}
    </div>
  );
};
