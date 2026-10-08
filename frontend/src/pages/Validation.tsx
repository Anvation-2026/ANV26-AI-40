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

  const chartData = report?.split_counts
    ? [
        { name: 'Train Set', count: report.split_counts.train || 0 },
        { name: 'Validation Set', count: report.split_counts.validation ?? (report.split_counts as any).val ?? 0 },
        { name: 'Test Set', count: report.split_counts.test || 0 },
      ]
    : [];

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-navy-foreground tracking-tight">
            Benchmark Validation Dashboard
          </h1>
          <p className="text-xs text-navy-muted mt-0.5 max-w-2xl">
            Empirical diagnostic metrics loaded directly from the validation report file. No values are fabricated or estimated.
          </p>
        </div>

        <button
          type="button"
          onClick={fetchValidation}
          disabled={loading}
          aria-label="Refresh validation metrics"
          className="self-start sm:self-auto inline-flex items-center gap-1.5 px-3 py-2 bg-surface hover:bg-slate-50 text-navy-foreground text-xs font-semibold rounded-[8px] border border-border shadow-2xs transition-colors focus:outline-none focus:ring-2 focus:ring-teal-600"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-teal-700' : ''}`} />
          <span>Refresh Benchmark</span>
        </button>
      </div>

      {loading && !data && (
        <div className="bg-surface rounded-[12px] border border-border p-16 text-center shadow-xs">
          <div className="w-8 h-8 rounded-full border-3 border-teal-200 border-t-teal-700 animate-spin mx-auto mb-3"></div>
          <p className="text-xs font-semibold text-navy-foreground">Loading Validation Metrics...</p>
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
        <div className="bg-surface rounded-[12px] border border-amber-200 p-10 text-center shadow-xs max-w-2xl mx-auto">
          <div className="w-12 h-12 rounded-[10px] bg-amber-50 text-amber-600 flex items-center justify-center mx-auto mb-3 border border-amber-100">
            <Clock className="w-6 h-6" />
          </div>
          <h2 className="text-base font-bold text-navy-foreground">
            Validation Benchmark Pending
          </h2>
          <p className="text-xs text-navy-muted mt-1.5 max-w-md mx-auto leading-relaxed">
            The machine learning evaluation report has not been generated yet. Metrics will appear automatically once the AI evaluation script produces the output report.
          </p>
          <div className="mt-4 inline-flex items-center gap-1.5 text-[11px] text-[#92400E] bg-[#FFFBEB] px-3 py-1.5 rounded-full border border-amber-200">
            <Info className="w-3.5 h-3.5" />
            <span>Target Path: ml/reports/validation_report.json</span>
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
        <div className="space-y-6">
          {/* Metadata Bar */}
          <div className="bg-surface rounded-[12px] border border-border p-4 shadow-xs flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-4 flex-wrap">
              <div>
                <span className="text-[10px] font-semibold text-navy-muted uppercase tracking-wider block">
                  Model Architecture
                </span>
                <span className="text-xs font-bold text-navy-foreground">
                  {report.model_name || 'ResNet-18'} ({report.model_version || 'v1'})
                </span>
              </div>

              <div className="h-6 w-px bg-border hidden sm:block"></div>

              <div>
                <span className="text-[10px] font-semibold text-navy-muted uppercase tracking-wider block">
                  Dataset Split
                </span>
                <span className="text-xs font-bold text-navy-foreground">
                  {report.dataset || 'PneumoniaMNIST+'} ({report.evaluated_on || 'test partition'})
                </span>
              </div>
            </div>

            <div>
              <span className="text-[10px] font-semibold text-navy-muted uppercase tracking-wider block text-right">
                Report Timestamp
              </span>
              <span className="text-xs font-mono tabular-nums text-navy-foreground">
                {formatDate(report.generated_at)}
              </span>
            </div>
          </div>

          {/* Dataset Splits Chart */}
          {chartData.length > 0 && (
            <div className="bg-surface rounded-[12px] border border-border p-5 shadow-xs">
              <div className="mb-3">
                <h3 className="text-sm font-bold text-navy-foreground tracking-tight">
                  Sample Distribution Across Partitions
                </h3>
                <span className="text-xs text-navy-muted">
                  Total sample counts in PneumoniaMNIST+ dataset
                </span>
              </div>
              <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E3E9EF" />
                    <XAxis dataKey="name" stroke="#617286" fontSize={11} tickLine={false} />
                    <YAxis stroke="#617286" fontSize={11} tickLine={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#FFFFFF',
                        border: '1px solid #E3E9EF',
                        borderRadius: '8px',
                        fontSize: '11px',
                        boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.05)',
                      }}
                    />
                    <Bar dataKey="count" fill="#0F766E" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Performance Metrics Grid */}
          <div>
            <div className="mb-3">
              <h3 className="text-sm font-bold text-navy-foreground tracking-tight">
                Statistical Performance & Discrimination
              </h3>
              <span className="text-xs text-navy-muted">
                Validation metrics evaluated on held-out test partition
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
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
                description="Positive predictive reliability for pneumonia"
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
                description="Difference between confidence and actual accuracy"
              />
              <MetricCard
                label="Abstention Coverage"
                value={report.metrics?.abstention_coverage}
                description="Percentage of samples with high confidence"
              />
              <MetricCard
                label="Rejection Rate"
                value={report.metrics?.rejection_rate}
                description="Rate of degraded or ambiguous samples rejected"
              />
            </div>
          </div>

          {/* Confusion Matrix */}
          <ConfusionMatrix data={report.confusion_matrix} />

          {/* Documented Model Limitations */}
          {report.limitations.length > 0 && (
            <div className="bg-surface rounded-[12px] border border-border p-5 shadow-xs">
              <h3 className="text-sm font-bold text-navy-foreground tracking-tight mb-1.5 flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-clinical-warning" />
                <span>Documented Model Limitations & Clinical Boundaries</span>
              </h3>
              <ul className="space-y-1.5 mt-2">
                {report.limitations.map((item, idx) => (
                  <li
                    key={idx}
                    className="text-xs text-navy-muted list-disc list-inside leading-relaxed"
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
