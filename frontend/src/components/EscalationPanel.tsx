import React from 'react';
import { UserCheck, ShieldCheck } from 'lucide-react';
import { TriageInfo } from '../types/analysis';

interface EscalationPanelProps {
  triage: TriageInfo;
}

export const EscalationPanel: React.FC<EscalationPanelProps> = ({ triage }) => {
  return (
    <div className="bg-navy text-white rounded-[12px] p-5 shadow-xs">
      <div className="flex items-start gap-3.5">
        <div className="p-2.5 bg-teal-800/60 rounded-[10px] text-teal-300 flex-shrink-0 mt-0.5">
          <UserCheck className="w-5 h-5" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 bg-teal-900/80 text-teal-300 rounded border border-teal-700/50">
              Triage Protocol
            </span>
            <span className="text-[11px] text-slate-300 font-medium">
              Educational Decision Support
            </span>
          </div>

          <h3 className="text-sm sm:text-base font-bold text-white tracking-tight">
            {triage.title}
          </h3>

          <p className="text-xs text-slate-300 mt-1 leading-relaxed">
            {triage.message}
          </p>

          {triage.reasons.length > 0 && (
            <div className="mt-3 bg-[#0A1724] p-3 rounded-[8px] border border-white/5">
              <span className="text-[11px] font-semibold text-teal-400 block mb-1">
                Trigger Factors:
              </span>
              <ul className="space-y-1">
                {triage.reasons.map((reason, idx) => (
                  <li key={idx} className="text-[11px] text-slate-300 list-disc list-inside">
                    {reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="mt-3 pt-2.5 border-t border-white/10 flex items-center gap-2 text-[11px] text-teal-300 font-medium">
            <ShieldCheck className="w-4 h-4 flex-shrink-0 text-teal-400" />
            <span>Human expert interpretation is strictly required for clinical verification.</span>
          </div>
        </div>
      </div>
    </div>
  );
};
