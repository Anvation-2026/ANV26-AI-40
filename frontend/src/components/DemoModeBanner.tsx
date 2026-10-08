import React from 'react';
import { FlaskConical } from 'lucide-react';

interface DemoModeBannerProps {
  scenario?: string;
  onScenarioChange?: (scenario: string) => void;
}

export const DemoModeBanner: React.FC<DemoModeBannerProps> = ({
  scenario,
  onScenarioChange,
}) => {
  return (
    <div
      role="alert"
      className="mb-6 bg-amber-50 border-2 border-amber-400 text-amber-900 rounded-xl p-4 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4"
    >
      <div className="flex items-start gap-3">
        <div className="p-2 bg-amber-200/80 rounded-lg text-amber-900 mt-0.5">
          <FlaskConical className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-sm tracking-wide uppercase px-2 py-0.5 bg-amber-200 text-amber-950 rounded text-[11px]">
              Demo Mode Active
            </span>
            <span className="text-sm font-semibold">
              Synthetic Fixtures Only — Not Real ML Inference
            </span>
          </div>
          <p className="text-xs text-amber-800 mt-1">
            The backend is currently running with synthetic test fixtures. No real clinical model predictions or heatmaps are generated in this mode.
          </p>
        </div>
      </div>

      {onScenarioChange && (
        <div className="flex items-center gap-2 self-start md:self-auto bg-white/70 p-1.5 rounded-lg border border-amber-300">
          <label htmlFor="demo-scenario-select" className="text-xs font-semibold text-amber-950 whitespace-nowrap">
            Scenario Fixture:
          </label>
          <select
            id="demo-scenario-select"
            value={scenario || 'uncertain'}
            onChange={(e) => onScenarioChange(e.target.value)}
            className="text-xs bg-white border border-amber-300 rounded px-2 py-1 text-slate-800 font-medium focus:ring-2 focus:ring-amber-500 focus:outline-none"
          >
            <option value="uncertain">High Uncertainty (Abstained)</option>
            <option value="success_pneumonia">Positive (Pneumonia)</option>
            <option value="success_normal">Negative (Normal)</option>
            <option value="poor_quality">Poor Image Quality (Rejected)</option>
            <option value="ood">Out of Distribution (Rejected)</option>
          </select>
        </div>
      )}
    </div>
  );
};
