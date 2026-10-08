import React from 'react';
import { CheckCircle2, XCircle, MinusCircle, AlertTriangle } from 'lucide-react';
import { QualityInfo, OODInfo, QualityCheck } from '../types/analysis';

interface QualityPanelProps {
  quality: QualityInfo;
  ood: OODInfo;
}

export const QualityPanel: React.FC<QualityPanelProps> = ({ quality, ood }) => {
  const renderMetricRow = (label: string, check: QualityCheck | null | undefined) => {
    if (!check || check.value === null || check.value === undefined) {
      return (
        <tr className="border-b border-slate-100 last:border-0 text-xs">
          <td className="py-2.5 font-medium text-slate-700">{label}</td>
          <td className="py-2.5 text-slate-400">Not evaluated</td>
          <td className="py-2.5 text-slate-400">&mdash;</td>
          <td className="py-2.5 text-right">
            <span className="inline-flex items-center gap-1 text-slate-400 text-xs">
              <MinusCircle className="w-3.5 h-3.5" />
              <span>N/A</span>
            </span>
          </td>
        </tr>
      );
    }

    const passed = check.passed;

    return (
      <tr className="border-b border-slate-100 last:border-0 text-xs">
        <td className="py-2.5 font-medium text-slate-700">{label}</td>
        <td className="py-2.5 font-mono text-slate-700">{check.value.toFixed(1)}</td>
        <td className="py-2.5 font-mono text-slate-400">
          {check.threshold !== null ? `&ge; ${check.threshold.toFixed(1)}` : '&mdash;'}
        </td>
        <td className="py-2.5 text-right">
          {passed === true && (
            <span className="inline-flex items-center gap-1 text-emerald-600 font-medium">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Pass</span>
            </span>
          )}
          {passed === false && (
            <span className="inline-flex items-center gap-1 text-rose-600 font-medium">
              <XCircle className="w-3.5 h-3.5" />
              <span>Fail</span>
            </span>
          )}
          {passed === null && (
            <span className="inline-flex items-center gap-1 text-slate-400">
              <MinusCircle className="w-3.5 h-3.5" />
              <span>&mdash;</span>
            </span>
          )}
        </td>
      </tr>
    );
  };

  const qualityBadge = {
    acceptable: { bg: 'bg-emerald-50 text-emerald-800 border-emerald-300', label: 'Acceptable' },
    poor: { bg: 'bg-rose-50 text-rose-800 border-rose-300', label: 'Poor Quality' },
    not_evaluated: { bg: 'bg-slate-100 text-slate-700 border-slate-300', label: 'Not Evaluated' },
  }[quality.status];

  const oodLabel = ood.evaluated
    ? ood.is_ood
      ? 'Out of Distribution'
      : 'In-Distribution (X-Ray)'
    : 'Not Evaluated';

  const oodColor = ood.evaluated
    ? ood.is_ood
      ? 'bg-rose-50 text-rose-800 border-rose-300'
      : 'bg-emerald-50 text-emerald-800 border-emerald-300'
    : 'bg-slate-100 text-slate-700 border-slate-300';

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-bold text-slate-900 tracking-tight">
          Image Quality & Domain Check
        </h3>
        <div className="flex gap-2">
          <span className={`text-xs px-2.5 py-1 rounded-full border font-medium ${qualityBadge.bg}`}>
            {qualityBadge.label}
          </span>
          <span className={`text-xs px-2.5 py-1 rounded-full border font-medium ${oodColor}`}>
            {oodLabel}
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-slate-200 text-[11px] uppercase tracking-wider text-slate-400">
              <th className="pb-2 font-semibold">Check</th>
              <th className="pb-2 font-semibold">Value</th>
              <th className="pb-2 font-semibold">Threshold</th>
              <th className="pb-2 font-semibold text-right">Result</th>
            </tr>
          </thead>
          <tbody>
            {renderMetricRow('Sharpness / Blur', quality.blur)}
            {renderMetricRow('Brightness', quality.brightness)}
            {renderMetricRow('Contrast', quality.contrast)}
          </tbody>
        </table>
      </div>

      {quality.reasons.length > 0 && (
        <div className="mt-4 p-3 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800">
          <div className="flex items-center gap-1.5 font-semibold mb-1">
            <AlertTriangle className="w-4 h-4 text-rose-600" />
            <span>Quality Defect Reasons</span>
          </div>
          <ul className="list-disc list-inside space-y-0.5 text-[11px]">
            {quality.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      {ood.is_ood && (
        <div className="mt-3 p-3 bg-purple-50 border border-purple-200 rounded-xl text-xs text-purple-900">
          <span className="font-semibold block mb-0.5">Domain Guard Triggered</span>
          <span>{ood.reason || 'Image features deviate from chest X-ray distribution.'}</span>
        </div>
      )}
    </div>
  );
};
