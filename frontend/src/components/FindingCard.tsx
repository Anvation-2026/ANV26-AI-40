import React from 'react';
import { StatusBadge } from './StatusBadge';
import { AnalysisResponse } from '../types/analysis';
import { CheckCircle2, AlertTriangle, ImageOff, Compass, AlertOctagon } from 'lucide-react';

interface FindingCardProps {
  analysis: AnalysisResponse;
}

export const FindingCard: React.FC<FindingCardProps> = ({ analysis }) => {
  const isSuccess = analysis.status === 'success';

  if (!isSuccess) {
    // Non-success states: strictly suppress finding and probability
    return (
      <div className="bg-surface rounded-[12px] border border-border p-5 shadow-xs">
        <div className="flex items-center justify-between gap-3 mb-3.5 flex-wrap">
          <StatusBadge status={analysis.status} size="md" />
          <span className="text-[11px] font-mono text-navy-muted">
            REQ: {analysis.request_id.slice(0, 8)}
          </span>
        </div>

        <div className="p-4 rounded-[10px] bg-canvas border border-border flex items-start gap-3">
          {analysis.status === 'uncertain' && (
            <AlertTriangle className="w-5 h-5 text-clinical-warning flex-shrink-0 mt-0.5" />
          )}
          {analysis.status === 'poor_quality' && (
            <ImageOff className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
          )}
          {analysis.status === 'ood' && (
            <Compass className="w-5 h-5 text-purple-700 flex-shrink-0 mt-0.5" />
          )}
          {(analysis.status === 'model_unavailable' ||
            analysis.status === 'error' ||
            analysis.status === 'invalid_input') && (
            <AlertOctagon className="w-5 h-5 text-clinical-danger flex-shrink-0 mt-0.5" />
          )}

          <div>
            <h3 className="text-sm font-semibold text-navy-foreground">
              {analysis.triage.title}
            </h3>
            <p className="text-xs text-navy-muted mt-1 leading-relaxed">
              {analysis.triage.message}
            </p>
            {analysis.triage.reasons.length > 0 && (
              <ul className="mt-2 space-y-1">
                {analysis.triage.reasons.map((r, i) => (
                  <li key={i} className="text-[11px] text-navy-muted list-disc list-inside">
                    {r}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        <p className="mt-3 text-[11px] text-navy-muted italic">
          No definitive finding or probability is displayed for abstained or rejected inputs.
        </p>
      </div>
    );
  }

  // Success state
  const isPneumonia = analysis.finding === 'pneumonia';
  const findingLabel = isPneumonia ? 'Pneumonia Pattern Detected' : 'Normal / Unremarkable Lung Fields';
  const themeClasses = isPneumonia
    ? 'text-[#92400E] bg-[#FFFBEB] border-amber-200'
    : 'text-[#065F46] bg-[#ECFDF5] border-emerald-200';

  return (
    <div className="bg-surface rounded-[12px] border border-border p-5 shadow-xs">
      <div className="flex items-center justify-between gap-3 mb-3.5 flex-wrap">
        <StatusBadge status={analysis.status} size="md" />
        <span className="text-[11px] font-mono text-navy-muted">
          REQ: {analysis.request_id.slice(0, 8)}
        </span>
      </div>

      <div className="mt-1">
        <span className="text-[11px] uppercase tracking-wider font-semibold text-navy-muted block mb-1.5">
          Model Finding (Educational)
        </span>
        <div className={`p-4 rounded-[10px] border flex items-center justify-between gap-4 ${themeClasses}`}>
          <div>
            <h2 className="text-lg font-bold tracking-tight">
              {findingLabel}
            </h2>
            <p className="text-xs mt-0.5 opacity-90">
              Statistical class alignment for instructional review &middot; Non-diagnostic
            </p>
          </div>
          <CheckCircle2 className="w-6 h-6 flex-shrink-0 opacity-90" />
        </div>
      </div>
    </div>
  );
};
