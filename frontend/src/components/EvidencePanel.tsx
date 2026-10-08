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
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-5">
      <div>
        <h3 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2 mb-2">
          <FileText className="w-4 h-4 text-teal-600" />
          <span>Clinical & Technical Explanation</span>
        </h3>
        <p className="text-xs sm:text-sm text-slate-700 leading-relaxed bg-slate-50 p-3.5 rounded-xl border border-slate-200">
          {explanation}
        </p>
      </div>

      {evidence.length > 0 && (
        <div>
          <h4 className="text-xs uppercase tracking-wider font-semibold text-slate-500 mb-2">
            Supporting Evidence Factors
          </h4>
          <ul className="space-y-1.5">
            {evidence.map((item, idx) => (
              <li
                key={idx}
                className="text-xs text-slate-700 flex items-start gap-2 bg-slate-50/70 p-2 rounded-lg border border-slate-100"
              >
                <Check className="w-3.5 h-3.5 text-teal-600 flex-shrink-0 mt-0.5" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Collapsible System Limitations */}
      <div className="pt-2 border-t border-slate-100">
        <button
          type="button"
          onClick={() => setLimitationsOpen(!limitationsOpen)}
          className="w-full flex items-center justify-between text-xs font-semibold text-slate-600 hover:text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-500 rounded p-1"
        >
          <span className="flex items-center gap-1.5">
            <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
            <span>Prototype Limitations & Boundary Constraints ({limitations.length})</span>
          </span>
          {limitationsOpen ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </button>

        {limitationsOpen && (
          <ul className="mt-3 space-y-1.5 pl-2">
            {limitations.map((limit, idx) => (
              <li
                key={idx}
                className="text-xs text-slate-500 list-disc list-inside leading-relaxed"
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
