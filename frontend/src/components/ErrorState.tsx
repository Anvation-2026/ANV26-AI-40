import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  showCommandHelp?: boolean;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Analysis Notice',
  message,
  onRetry,
  showCommandHelp = true,
}) => {
  return (
    <div className="bg-surface rounded-[12px] border border-border p-5 shadow-xs text-center max-w-md mx-auto">
      <div className="w-10 h-10 rounded-[8px] bg-rose-50 text-clinical-danger flex items-center justify-center mx-auto mb-2.5 border border-rose-200">
        <AlertCircle className="w-5 h-5" />
      </div>

      <h3 className="text-sm font-bold text-navy-foreground tracking-tight">
        {title}
      </h3>

      <p className="text-xs text-navy-muted mt-1 leading-relaxed">
        {message}
      </p>

      {showCommandHelp && (
        <div className="mt-3 p-2.5 bg-canvas rounded-[6px] border border-border text-left font-mono text-[11px] text-navy-muted">
          <code>uvicorn backend.main:app --port 8000</code>
        </div>
      )}

      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 inline-flex items-center justify-center gap-1.5 px-4 py-2 bg-teal-700 hover:bg-teal-800 text-white text-xs font-semibold rounded-[8px] transition-colors focus:outline-none focus:ring-2 focus:ring-teal-600"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Retry Analysis</span>
        </button>
      )}
    </div>
  );
};
