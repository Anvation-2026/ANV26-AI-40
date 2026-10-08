import React, { useEffect, useState } from 'react';
import {
  Cpu,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  FlaskConical,
  XCircle,
} from 'lucide-react';
import { getModelStatus } from '../services/api';
import { ModelStatusResponse } from '../types/analysis';

interface ModelStatusCardProps {
  onStatusLoaded?: (status: ModelStatusResponse | null) => void;
  compact?: boolean;
}

export const ModelStatusCard: React.FC<ModelStatusCardProps> = ({
  onStatusLoaded,
  compact = false,
}) => {
  const [status, setStatus] = useState<ModelStatusResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = () => {
    setLoading(true);
    setError(null);
    getModelStatus()
      .then((data) => {
        setStatus(data);
        setError(null);
        setLoading(false);
        if (onStatusLoaded) onStatusLoaded(data);
      })
      .catch((err) => {
        setError(err.message || 'API server offline');
        setStatus(null);
        setLoading(false);
        if (onStatusLoaded) onStatusLoaded(null);
      });
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  if (compact) {
    return (
      <div className="flex items-center gap-2 text-xs">
        {loading ? (
          <span className="inline-flex items-center gap-1.5 text-navy-muted">
            <RefreshCw className="w-3 h-3 animate-spin text-teal-700" />
            <span>Checking engine...</span>
          </span>
        ) : error ? (
          <div className="inline-flex items-center gap-2">
            <span className="inline-flex items-center gap-1 text-clinical-danger font-medium bg-rose-50 px-2 py-0.5 rounded border border-rose-200 text-[11px]">
              <XCircle className="w-3 h-3" />
              <span>Offline</span>
            </span>
            <button
              onClick={fetchStatus}
              className="text-[11px] text-teal-700 hover:underline inline-flex items-center gap-1 font-medium"
            >
              <RefreshCw className="w-2.5 h-2.5" />
              <span>Retry</span>
            </button>
          </div>
        ) : status?.mode === 'demo' ? (
          <span className="inline-flex items-center gap-1 text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-200 text-[11px] font-semibold">
            <FlaskConical className="w-3 h-3" />
            <span>Demo Mode</span>
          </span>
        ) : status?.inference_ready ? (
          <span className="inline-flex items-center gap-1 text-clinical-success bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 text-[11px] font-semibold">
            <CheckCircle2 className="w-3 h-3" />
            <span>Engine Ready ({status.model_version || 'v1'})</span>
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 text-slate-700 bg-slate-100 px-2 py-0.5 rounded border border-slate-200 text-[11px] font-medium">
            <AlertTriangle className="w-3 h-3 text-amber-600" />
            <span>Model Unloaded</span>
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="bg-surface rounded-[12px] border border-border px-4 py-3 shadow-xs flex flex-wrap items-center justify-between gap-3 text-xs">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-[8px] bg-canvas border border-border flex items-center justify-center text-teal-700 flex-shrink-0">
          <Cpu className="w-4 h-4" />
        </div>

        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-navy-foreground text-xs">
              Model Engine Status:
            </span>

            {loading ? (
              <span className="text-navy-muted inline-flex items-center gap-1">
                <RefreshCw className="w-3 h-3 animate-spin text-teal-700" />
                <span>Connecting...</span>
              </span>
            ) : error ? (
              <span className="text-clinical-danger font-semibold inline-flex items-center gap-1">
                <XCircle className="w-3.5 h-3.5" />
                <span>Backend Offline</span>
              </span>
            ) : status?.mode === 'demo' ? (
              <span className="text-amber-800 font-semibold inline-flex items-center gap-1">
                <FlaskConical className="w-3.5 h-3.5 text-amber-600" />
                <span>Demo Fixture Mode</span>
              </span>
            ) : status?.inference_ready ? (
              <span className="text-clinical-success font-semibold inline-flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Active & Ready</span>
              </span>
            ) : (
              <span className="text-navy-muted inline-flex items-center gap-1">
                <AlertTriangle className="w-3.5 h-3.5 text-clinical-warning" />
                <span>Model Unavailable (Contract Standby)</span>
              </span>
            )}
          </div>

          <p className="text-[11px] text-navy-muted mt-0.5">
            {error
              ? 'Start backend: uvicorn backend.main:app --port 8000'
              : status?.message ||
                `Architecture: ${status?.model_name || 'ResNet-18'} · Dataset: PneumoniaMNIST+`}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={fetchStatus}
          disabled={loading}
          className="inline-flex items-center gap-1 px-2.5 py-1.5 text-[11px] font-medium text-navy-muted hover:text-navy-foreground bg-canvas hover:bg-slate-200/70 border border-border rounded-[6px] transition-colors focus:outline-none focus:ring-2 focus:ring-teal-600"
          title="Refresh connection status"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin text-teal-700' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>
    </div>
  );
};
