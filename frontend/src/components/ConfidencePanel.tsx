import React from 'react';
import { HelpCircle } from 'lucide-react';
import { AnalysisResponse } from '../types/analysis';
import { formatProbability, formatScore } from '../lib/format';

interface ConfidencePanelProps {
  analysis: AnalysisResponse;
}

export const ConfidencePanel: React.FC<ConfidencePanelProps> = ({ analysis }) => {
  const {
    probability,
    probability_of,
    raw_score,
    uncertainty,
    abstained,
    status,
  } = analysis;

  const isSuccess = status === 'success';

  // Uncertainty badge color
  const uncertaintyBadge = {
    low: { bg: 'bg-emerald-50 text-emerald-800 border-emerald-300', label: 'Low Uncertainty' },
    moderate: { bg: 'bg-amber-50 text-amber-800 border-amber-300', label: 'Moderate Uncertainty' },
    high: { bg: 'bg-rose-50 text-rose-800 border-rose-300', label: 'High Uncertainty' },
    not_evaluated: { bg: 'bg-slate-100 text-slate-700 border-slate-300', label: 'Not Evaluated' },
  }[uncertainty.level];

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2">
          Model Confidence & Uncertainty
        </h3>
        <span
          className={`text-xs px-2.5 py-1 rounded-full border font-medium ${uncertaintyBadge.bg}`}
        >
          {uncertaintyBadge.label}
        </span>
      </div>

      {isSuccess && probability !== null ? (
        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-baseline mb-1">
              <span className="text-xs font-semibold text-slate-600">
                Calibrated Probability {probability_of ? `(of ${probability_of})` : ''}
              </span>
              <span className="text-2xl font-black text-teal-600">
                {formatProbability(probability)}
              </span>
            </div>
            {/* Progress bar strictly when non-null */}
            <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
              <div
                className="bg-teal-600 h-2.5 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, Math.max(0, probability * 100))}%` }}
              ></div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-100 text-xs">
            <div>
              <span className="text-slate-400 block">Raw Model Score</span>
              <span className="font-semibold text-slate-700 font-mono">
                {formatScore(raw_score)}
              </span>
            </div>
            <div>
              <span className="text-slate-400 block">Uncertainty Metric</span>
              <span className="font-semibold text-slate-700 font-mono">
                {uncertainty.value !== null ? uncertainty.value.toFixed(3) : 'Not evaluated'}
              </span>
              {uncertainty.method && (
                <span className="text-[10px] text-slate-400 block">
                  {uncertainty.method}
                </span>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="py-4 text-center bg-slate-50 rounded-xl border border-slate-200">
          <p className="text-xs font-medium text-slate-500">
            Probability metric: <span className="font-semibold">Not evaluated / Suppressed</span>
          </p>
          {abstained && (
            <p className="text-[11px] text-amber-700 mt-1">
              Safety abstention active &middot; Model did not commit to confidence output
            </p>
          )}
        </div>
      )}

      {/* Mandatory clinical safety disclaimer */}
      <div className="mt-4 pt-3 border-t border-slate-100 flex items-start gap-1.5 text-[11px] text-slate-500">
        <HelpCircle className="w-3.5 h-3.5 text-slate-400 flex-shrink-0 mt-0.5" />
        <span>
          High probability does not mean clinical reliability. Calibrated probabilities reflect statistical dataset alignment, not verified medical pathology.
        </span>
      </div>
    </div>
  );
};
