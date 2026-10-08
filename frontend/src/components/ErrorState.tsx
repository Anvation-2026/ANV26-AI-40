import React from 'react';
import { AlertOctagon, RefreshCw, Terminal } from 'lucide-react';

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  showCommandHelp?: boolean;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Connection or Analysis Error',
  message,
  onRetry,
  showCommandHelp = true,
}) => {
  return (
    <div className="bg-white rounded-2xl border border-rose-200 p-8 shadow-xs text-center max-w-xl mx-auto my-8">
      <div className="w-14 h-14 rounded-2xl bg-rose-50 text-rose-600 flex items-center justify-center mx-auto mb-4 border border-rose-100">
        <AlertOctagon className="w-7 h-7" />
      </div>

      <h3 className="text-lg font-bold text-slate-900 tracking-tight">
        {title}
      </h3>

      <p className="text-sm text-slate-600 mt-2 leading-relaxed">
        {message}
      </p>

      {showCommandHelp && (
        <div className="mt-5 p-3.5 bg-slate-900 text-slate-200 rounded-xl text-left font-mono text-xs overflow-x-auto shadow-inner">
          <div className="flex items-center gap-1.5 text-slate-400 mb-1 text-[11px]">
            <Terminal className="w-3.5 h-3.5" />
            <span>Start backend server:</span>
          </div>
          <code>uvicorn backend.main:app --reload --port 8000</code>
        </div>
      )}

      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-6 inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-teal-600 hover:bg-teal-700 text-white text-sm font-semibold rounded-xl transition-colors focus:outline-none focus:ring-4 focus:ring-teal-500/20"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Retry Operation</span>
        </button>
      )}
    </div>
  );
};
