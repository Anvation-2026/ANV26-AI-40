import React from 'react';
import { UserCheck, ShieldCheck } from 'lucide-react';
import { TriageInfo } from '../types/analysis';

interface EscalationPanelProps {
  triage: TriageInfo;
}

export const EscalationPanel: React.FC<EscalationPanelProps> = ({ triage }) => {
  return (
    <div className="bg-teal-900 text-white rounded-2xl p-6 shadow-sm">
      <div className="flex items-start gap-3.5">
        <div className="p-2.5 bg-teal-800 rounded-xl text-teal-200 flex-shrink-0">
          <UserCheck className="w-6 h-6" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs uppercase tracking-wider font-bold px-2 py-0.5 bg-teal-800 text-teal-200 rounded">
              Triage Protocol
            </span>
            <span className="text-xs text-teal-300 font-medium">
              Educational Oversight
            </span>
          </div>

          <h3 className="text-base font-bold text-white mt-1.5">
            {triage.title}
          </h3>

          <p className="text-sm text-teal-100 mt-1 leading-relaxed">
            {triage.message}
          </p>

          {triage.reasons.length > 0 && (
            <div className="mt-3 bg-teal-950/60 p-3 rounded-xl border border-teal-800/80">
              <span className="text-xs font-semibold text-teal-300 block mb-1">
                Triage Factors:
              </span>
              <ul className="space-y-1">
                {triage.reasons.map((reason, idx) => (
                  <li key={idx} className="text-xs text-teal-200 list-disc list-inside">
                    {reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="mt-4 pt-3 border-t border-teal-800/60 flex items-center gap-2 text-xs text-teal-200 font-medium">
            <ShieldCheck className="w-4 h-4 text-teal-300 flex-shrink-0" />
            <span>Human expert interpretation is always strictly required.</span>
          </div>
        </div>
      </div>
    </div>
  );
};
