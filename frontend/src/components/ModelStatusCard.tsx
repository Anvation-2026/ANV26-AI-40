import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Cpu,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  FlaskConical,
  XCircle,
  ArrowRight,
} from 'lucide-react';
import { getModelStatus } from '../services/api';
import { ModelStatusResponse } from '../types/analysis';

export const ModelStatusCard: React.FC = () => {
  const [status, setStatus] = useState<ModelStatusResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = () => {
    setLoading(true);
    setError(null);
    getModelStatus()
      .then((data) => {
        setStatus(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to reach API server');
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-teal-50 text-teal-600 rounded-xl">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900 tracking-tight">
              ML Decision Support Engine Status
            </h3>
            <span className="text-xs text-slate-500">Live operational readiness</span>
          </div>
        </div>

        <button
          type="button"
          onClick={fetchStatus}
          disabled={loading}
          aria-label="Refresh model status"
          className="p-1.5 hover:bg-slate-100 rounded-lg text-slate-400 hover:text-slate-600 transition-colors focus:outline-none focus:ring-2 focus:ring-teal-500"
          title="Refresh status"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-teal-600' : ''}`} />
        </button>
      </div>

      {loading && !status && !error && (
        <div className="py-8 text-center text-xs text-slate-500">
          Checking backend connection and model state...
        </div>
      )}

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800">
          <div className="flex items-center gap-2 font-semibold mb-1">
            <XCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span>Backend Unreachable</span>
          </div>
          <p>{error}</p>
          <p className="mt-2 text-rose-600">
            Make sure the FastAPI backend is running on <code>http://localhost:8000</code>.
          </p>
        </div>
      )}

      {status && !error && (
        <div>
          {/* Status Indicator */}
          <div className="flex items-center gap-2.5 mb-4 p-3 rounded-xl bg-slate-50 border border-slate-200">
            {status.mode === 'demo' ? (
              <>
                <FlaskConical className="w-5 h-5 text-amber-600 flex-shrink-0" />
                <div>
                  <span className="text-xs font-bold text-amber-900 block">
                    Demo Mode Active
                  </span>
                  <span className="text-[11px] text-amber-700">
                    Synthetic test fixtures active for UI verification.
                  </span>
                </div>
              </>
            ) : status.inference_ready ? (
              <>
                <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0" />
                <div>
                  <span className="text-xs font-bold text-emerald-900 block">
                    Inference Engine Ready
                  </span>
                  <span className="text-[11px] text-emerald-700">
                    Model weights loaded & ready for evaluation.
                  </span>
                </div>
              </>
            ) : (
              <>
                <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0" />
                <div>
                  <span className="text-xs font-bold text-amber-900 block">
                    Model Module Unavailable
                  </span>
                  <span className="text-[11px] text-amber-700">
                    {status.message || 'ML weights or module not yet integrated into environment.'}
                  </span>
                </div>
              </>
            )}
          </div>

          {/* Model Details Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs mb-6">
            <div className="p-2.5 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-slate-400 block text-[11px]">Model Architecture</span>
              <span className="font-semibold text-slate-800 truncate block">
                {status.model_name || 'Not integrated'}
              </span>
            </div>
            <div className="p-2.5 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-slate-400 block text-[11px]">Checkpoint Version</span>
              <span className="font-semibold text-slate-800 truncate block font-mono">
                {status.model_version || 'N/A'}
              </span>
            </div>
            <div className="p-2.5 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-slate-400 block text-[11px]">Supported Domain</span>
              <span className="font-semibold text-slate-800 truncate block">
                {status.supported_image_type}
              </span>
            </div>
            <div className="p-2.5 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-slate-400 block text-[11px]">Runtime Mode</span>
              <span className="font-semibold text-slate-800 uppercase tracking-wider text-[11px] block">
                {status.mode}
              </span>
            </div>
          </div>

          {/* Call to Action Buttons */}
          <div className="flex flex-col sm:flex-row items-center gap-3 pt-2">
            <Link
              to="/analyze"
              className="w-full sm:w-auto flex-1 inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-teal-600 hover:bg-teal-700 text-white text-sm font-semibold rounded-xl shadow-xs transition-colors focus:outline-none focus:ring-4 focus:ring-teal-500/20"
            >
              <span>Analyze Chest X-Ray</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/validation"
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-4 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold rounded-xl transition-colors focus:outline-none focus:ring-4 focus:ring-slate-300/40"
            >
              <span>Validation Dashboard</span>
            </Link>
          </div>
        </div>
      )}
    </div>
  );
};
