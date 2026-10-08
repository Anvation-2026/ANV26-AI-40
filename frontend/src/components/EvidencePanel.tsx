import React, { useState } from 'react';
import { ChevronDown, ChevronUp, FileText, Check, AlertCircle } from 'lucide-react';

interface EvidencePanelProps {
  explanation: string;
  evidence: string[];
  limitations: string[];
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  explanation,
  evidence,
  limitations,
}) => {
  const [limitationsOpen, setLimitationsOpen] = useState<boolean>(false);

  return (
    <div className="bg-surface rounded-[12px] border border-border p-5 shadow-xs space-y-4">
      <div>
        <h3 className="text-sm font-bold text-navy-foreground tracking-tight flex items-center gap-2 mb-1.5">
          <FileText className="w-4 h-4 text-teal-700" />
          <span>Decision Context & Explanation</span>
        </h3>
        <p className="text-xs text-navy-foreground leading-relaxed bg-canvas p-3.5 rounded-[10px] border border-border">
          {explanation}
        </p>
      </div>

      {evidence.length > 0 && (
        <div>
          <h4 className="text-[11px] uppercase tracking-wider font-semibold text-navy-muted mb-2">
            Model-Grounded Evidence Observations
          </h4>
          <ul className="space-y-1.5">
            {evidence.map((item, idx) => {
              const isPossibility = item.toLowerCase().includes('possibility evaluated') || item.toLowerCase().includes('differential consideration');
              return (
                <li
                  key={idx}
                  className={`text-xs flex items-start gap-2 p-2 rounded-[8px] border ${
                    isPossibility
                      ? 'bg-teal-50/50 border-teal-200/70 text-navy-foreground'
                      : 'bg-canvas border-border/60 text-navy-foreground'
                  }`}
                >
                  <Check className={`w-3.5 h-3.5 flex-shrink-0 mt-0.5 ${isPossibility ? 'text-teal-800 font-bold' : 'text-teal-700'}`} />
                  <span className="leading-tight">{item}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* Collapsible System Limitations */}
      <div className="pt-2 border-t border-border/60">
        <button
          type="button"
          onClick={() => setLimitationsOpen(!limitationsOpen)}
          className="w-full flex items-center justify-between text-xs font-semibold text-navy-muted hover:text-navy-foreground focus:outline-none focus:ring-2 focus:ring-teal-600 rounded p-1 transition-colors"
        >
          <span className="flex items-center gap-1.5">
            <AlertCircle className="w-3.5 h-3.5 text-clinical-warning" />
            <span>Prototype Limitations & Boundary Constraints ({limitations.length})</span>
          </span>
          {limitationsOpen ? (
            <ChevronUp className="w-4 h-4 text-navy-muted" />
          ) : (
            <ChevronDown className="w-4 h-4 text-navy-muted" />
          )}
        </button>

        {limitationsOpen && (
          <ul className="mt-2.5 space-y-1.5 pl-2">
            {limitations.map((limit, idx) => (
              <li
                key={idx}
                className="text-[11px] text-navy-muted list-disc list-inside leading-relaxed"
              >
                {limit}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};
