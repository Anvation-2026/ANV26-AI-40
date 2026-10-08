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

  const uncertaintyConfig = {
    low: {
      bg: 'bg-emerald-50 text-emerald-800 border-emerald-200',
      label: 'Low Uncertainty',
      desc: 'Predictions show stable statistical convergence across MC-dropout iterations.',
    },
    moderate: {
      bg: 'bg-amber-50 text-amber-800 border-amber-200',
      label: 'Moderate Uncertainty',
      desc: 'Minor feature variance detected. Interpret results with heightened caution.',
    },
    high: {
      bg: 'bg-rose-50 text-rose-800 border-rose-200',
      label: 'High Uncertainty',
      desc: 'Substantial variance across prediction samples. Decision safety abstention triggered.',
    },
    not_evaluated: {
      bg: 'bg-slate-100 text-slate-700 border-slate-200',
      label: 'Not Evaluated',
      desc: 'Uncertainty estimation was not computed by the model engine.',
    },
  }[uncertainty.level];

  return (
    <div className="bg-surface rounded-[12px] border border-border p-5 shadow-xs space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-navy-foreground tracking-tight">
            Confidence & Uncertainty Calibration
          </h3>
          <span className="text-[11px] text-navy-muted">
            Probability scaling and ensemble variance metrics
          </span>
        </div>
        <span
          className={`text-[11px] px-2.5 py-1 rounded-full border font-semibold ${uncertaintyConfig.bg}`}
        >
          {uncertaintyConfig.label}
        </span>
      </div>

      {isSuccess && probability !== null ? (
        <div className="space-y-3.5">
          <div>
            <div className="flex justify-between items-baseline mb-1.5">
              <span className="text-xs font-semibold text-navy-foreground">
                Calibrated Probability {probability_of ? `(of ${probability_of})` : ''}
              </span>
              <span className="text-2xl font-black text-teal-700 font-mono tabular-nums">
                {formatProbability(probability)}
              </span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden border border-border/50">
              <div
                className="bg-teal-700 h-2 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, Math.max(0, probability * 100))}%` }}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-2.5 border-t border-border/60 text-xs">
            <div className="p-2.5 bg-canvas rounded-[8px] border border-border/60">
              <span className="text-navy-muted block text-[11px]">Raw Model Logit/Score</span>
              <span className="font-bold text-navy-foreground font-mono tabular-nums text-sm">
                {formatScore(raw_score)}
              </span>
            </div>
            <div className="p-2.5 bg-canvas rounded-[8px] border border-border/60">
              <span className="text-navy-muted block text-[11px]">Uncertainty Index</span>
              <span className="font-bold text-navy-foreground font-mono tabular-nums text-sm">
                {uncertainty.value !== null ? uncertainty.value.toFixed(3) : 'Not evaluated'}
              </span>
              {uncertainty.method && (
                <span className="text-[10px] text-navy-muted block truncate mt-0.5">
                  {uncertainty.method}
                </span>
              )}
            </div>
          </div>

          <p className="text-[11px] text-navy-muted leading-relaxed">
            {uncertaintyConfig.desc}
          </p>
        </div>
      ) : (
        <div className="py-4 px-3 text-center bg-canvas rounded-[10px] border border-border">
          <p className="text-xs font-medium text-navy-muted">
            Probability Metric: <strong className="text-navy-foreground">Not evaluated / Suppressed</strong>
          </p>
          {abstained && (
            <p className="text-[11px] text-clinical-warning font-medium mt-1">
              Safety Abstention Policy Active &middot; Model did not commit to confidence output
            </p>
          )}
        </div>
      )}

      {/* Safety Notice */}
      <div className="pt-2 border-t border-border/60 flex items-start gap-1.5 text-[11px] text-navy-muted leading-tight">
        <HelpCircle className="w-3.5 h-3.5 text-navy-muted flex-shrink-0 mt-0.5" />
        <span>
          High statistical probability does not constitute clinical certainty. Calibrated probabilities reflect dataset convergence, not verified medical status.
        </span>
      </div>
    </div>
  );
};
