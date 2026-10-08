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
    // Non-success states: show NO finding text and NO probability
    return (
      <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
        <div className="flex items-center justify-between gap-4 mb-4 flex-wrap">
          <StatusBadge status={analysis.status} size="lg" />
          <span className="text-xs font-mono text-slate-400">
            ID: {analysis.request_id.slice(0, 8)}...
          </span>
        </div>

        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex items-start gap-3.5">
          {analysis.status === 'uncertain' && (
            <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
          )}
          {analysis.status === 'poor_quality' && (
            <ImageOff className="w-5 h-5 text-orange-600 flex-shrink-0 mt-0.5" />
          )}
          {analysis.status === 'ood' && (
            <Compass className="w-5 h-5 text-purple-600 flex-shrink-0 mt-0.5" />
          )}
          {(analysis.status === 'model_unavailable' ||
            analysis.status === 'error' ||
            analysis.status === 'invalid_input') && (
            <AlertOctagon className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
          )}

          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              {analysis.triage.title}
            </h3>
            <p className="text-sm text-slate-600 mt-1">
              {analysis.triage.message}
            </p>
            {analysis.triage.reasons.length > 0 && (
              <ul className="mt-2 space-y-1">
                {analysis.triage.reasons.map((r, i) => (
                  <li key={i} className="text-xs text-slate-500 list-disc list-inside">
                    {r}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        <p className="mt-4 text-xs text-slate-500 italic">
          No definitive finding or probability is displayed for abstained or rejected inputs.
        </p>
      </div>
    );
  }

  // Success state
  const findingLabel = analysis.finding === 'pneumonia' ? 'Pneumonia Pattern' : 'Normal / Unremarkable';
  const findingColor =
    analysis.finding === 'pneumonia'
      ? 'text-amber-800 bg-amber-50 border-amber-300'
      : 'text-emerald-800 bg-emerald-50 border-emerald-300';

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
      <div className="flex items-center justify-between gap-4 mb-4 flex-wrap">
        <StatusBadge status={analysis.status} size="lg" />
        <span className="text-xs font-mono text-slate-400">
          ID: {analysis.request_id.slice(0, 8)}...
        </span>
      </div>

      <div className="mt-2">
        <span className="text-xs uppercase tracking-wider font-semibold text-slate-500 block">
          Model Finding (Educational)
        </span>
        <div className={`mt-2 p-4 rounded-xl border flex items-center justify-between gap-4 ${findingColor}`}>
          <div>
            <h2 className="text-xl font-bold tracking-tight">
              {findingLabel}
            </h2>
            <p className="text-xs mt-1 text-slate-600">
              Predicted classification for instructional review.
            </p>
          </div>
          <CheckCircle2 className="w-7 h-7 flex-shrink-0 opacity-80" />
        </div>
      </div>
    </div>
  );
};
