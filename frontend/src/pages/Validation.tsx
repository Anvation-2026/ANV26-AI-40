import React, { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import {
  Clock,
  Info,
  RefreshCw,
  ShieldAlert,
} from 'lucide-react';
import { MetricCard } from '../components/MetricCard';
import { ConfusionMatrix } from '../components/ConfusionMatrix';
import { ErrorState } from '../components/ErrorState';
import { getValidation } from '../services/api';
import { ValidationResponse } from '../types/analysis';
import { formatDate } from '../lib/format';

export const Validation: React.FC = () => {
  const [data, setData] = useState<ValidationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [networkError, setNetworkError] = useState<string | null>(null);

  const fetchValidation = () => {
    setLoading(true);
    setNetworkError(null);
    getValidation()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setNetworkError(err.message || 'Could not fetch validation metrics.');
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchValidation();
  }, []);

  const report = data?.report;

  // Split counts data formatted for Recharts
  const chartData = report?.split_counts
    ? [
        { name: 'Train', count: report.split_counts.train || 0 },
        { name: 'Validation', count: report.split_counts.validation || 0 },
        { name: 'Test', count: report.split_counts.test || 0 },
      ]
    : [];

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black text-slate-900 tracking-tight">
            Independent Model Validation Dashboard
          </h1>
          <p className="text-sm text-slate-500 mt-1 max-w-2xl">
            Live evaluation metrics derived strictly from the AI engineering validation report file. No values are fabricated or hardcoded.
          </p>
        </div>

        <button
          type="button"
          onClick={fetchValidation}
          disabled={loading}
          aria-label="Refresh validation metrics"
          className="self-start sm:self-auto inline-flex items-center gap-1.5 px-3.5 py-2 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold rounded-xl border border-slate-200 shadow-2xs transition-colors focus:outline-none focus:ring-2 focus:ring-teal-500"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-teal-600' : ''}`} />
          <span>Refresh Report</span>
        </button>
      </div>

      {loading && !data && (
        <div className="bg-white rounded-2xl border border-slate-200 p-16 text-center shadow-xs">
          <div className="w-10 h-10 rounded-full border-4 border-teal-200 border-t-teal-600 animate-spin mx-auto mb-3"></div>
          <p className="text-sm font-semibold text-slate-700">Loading Validation Metrics...</p>
        </div>
      )}

      {networkError && (
        <ErrorState
          title="Validation Fetch Failed"
          message={networkError}
          onRetry={fetchValidation}
        />
      )}

      {/* Pending State */}
      {!loading && data?.status === 'pending' && (
        <div className="bg-white rounded-2xl border border-amber-200 p-12 text-center shadow-xs max-w-2xl mx-auto">
          <div className="w-14 h-14 rounded-2xl bg-amber-50 text-amber-600 flex items-center justify-center mx-auto mb-4 border border-amber-100">
            <Clock className="w-7 h-7" />
          </div>
          <h2 className="text-lg font-bold text-slate-900">
            Evaluation Pending
          </h2>
          <p className="text-sm text-slate-600 mt-2 max-w-md mx-auto leading-relaxed">
            The ML validation report has not been produced yet by the AI engineering team. Check back once training and benchmark evaluation are complete.
          </p>
          <div className="mt-4 inline-flex items-center gap-1.5 text-xs text-amber-800 bg-amber-50 px-3 py-1.5 rounded-full border border-amber-200">
            <Info className="w-3.5 h-3.5" />
            <span>Expected location: ml/reports/validation_report.json</span>
          </div>
        </div>
      )}

      {/* Error State */}
      {!loading && data?.status === 'error' && (
        <ErrorState
          title="Malformed Validation Report"
          message={data.message || 'The validation report file could not be parsed into the required schema.'}
          onRetry={fetchValidation}
          showCommandHelp={false}
        />
      )}

      {/* Available State */}
      {!loading && data?.status === 'available' && report && (
        <div className="space-y-8">
          {/* Metadata Banner */}
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-4 flex-wrap">
              <div>
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                  Model
                </span>
                <span className="text-sm font-bold text-slate-800">
                  {report.model_name || 'Unnamed Model'} ({report.model_version || 'v1'})
                </span>
              </div>

              <div className="h-8 w-px bg-slate-200 hidden sm:block"></div>

              <div>
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                  Benchmark Dataset
                </span>
                <span className="text-sm font-bold text-slate-800">
                  {report.dataset || 'PneumoniaMNIST+'} ({report.evaluated_on || 'test partition'})
                </span>
              </div>
            </div>

            <div className="text-right">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                Report Generated
              </span>
              <span className="text-xs font-mono text-slate-600">
                {formatDate(report.generated_at)}
              </span>
            </div>
          </div>

          {/* Dataset Splits Chart */}
          {chartData.length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
              <div className="mb-4">
                <h3 className="text-sm font-bold text-slate-900 tracking-tight">
                  Benchmark Dataset Split Distribution
                </h3>
                <span className="text-xs text-slate-400">
                  Sample count partitioning across training, validation, and test sets
                </span>
              </div>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={12} tickLine={false} />
                    <YAxis stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#ffffff',
                        border: '1px solid #e2e8f0',
                        borderRadius: '0.75rem',
                        fontSize: '12px',
                        boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.1)',
                      }}
                    />
                    <Bar dataKey="count" fill="#0d9488" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Performance Metrics Grid */}
          <div>
            <h3 className="text-sm font-bold text-slate-900 tracking-tight mb-3">
              Diagnostic Performance & Calibration Metrics
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                label="Accuracy"
                value={report.metrics?.accuracy}
                description="Overall proportion of correct classifications"
              />
              <MetricCard
                label="Sensitivity / Recall"
                value={report.metrics?.sensitivity ?? report.metrics?.recall}
                description="True positive detection rate for pneumonia"
              />
              <MetricCard
                label="Specificity"
                value={report.metrics?.specificity}
                description="True negative identification rate for normal cases"
              />
              <MetricCard
                label="Precision (PPV)"
                value={report.metrics?.precision}
                description="Reliability of positive pneumonia predictions"
              />
              <MetricCard
                label="F1 Score"
                value={report.metrics?.f1}
                description="Harmonic mean of precision and sensitivity"
              />
              <MetricCard
                label="AUROC"
                value={report.metrics?.auroc}
                isPercentage={false}
                description="Area under ROC discrimination curve"
              />
              <MetricCard
                label="False Negative Rate"
                value={report.metrics?.false_negative_rate}
                description="Proportion of pneumonia cases missed"
              />
              <MetricCard
                label="Expected Calibration Error (ECE)"
                value={report.metrics?.ece}
                isPercentage={false}
                description="Discrepancy between confidence and accuracy"
              />
              <MetricCard
                label="Abstention Coverage"
                value={report.metrics?.abstention_coverage}
                description="Proportion of cases with high certainty"
              />
              <MetricCard
                label="Rejection Rate"
                value={report.metrics?.rejection_rate}
                description="Rate of abstained or poor-quality inputs"
              />
            </div>
          </div>

          {/* Confusion Matrix */}
          <ConfusionMatrix data={report.confusion_matrix} />

          {/* Documented Limitations */}
          {report.limitations.length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
              <h3 className="text-sm font-bold text-slate-900 tracking-tight mb-2 flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-amber-600" />
                <span>Documented Model Limitations & Clinical Boundaries</span>
              </h3>
              <ul className="space-y-1.5 mt-3">
                {report.limitations.map((item, idx) => (
                  <li
                    key={idx}
                    className="text-xs text-slate-600 list-disc list-inside leading-relaxed"
                  >
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
