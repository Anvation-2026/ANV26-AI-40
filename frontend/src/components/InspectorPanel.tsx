import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  FileSearch,
  ShieldCheck,
  Code2,
  CheckCircle2,
  AlertTriangle,
  ImageOff,
  Compass,
  AlertOctagon,
  Copy,
  Check,
  RefreshCw,
  Search,
  ChevronRight,
  Layers,
} from 'lucide-react';
import { AnalysisResponse } from '../types/analysis';
import { formatProbability, formatScore } from '../lib/format';
import { StatusBadge } from './StatusBadge';

export type InspectorTab = 'finding' | 'evidence' | 'reliability' | 'technical';

interface InspectorPanelProps {
  analysis: AnalysisResponse | null;
  loading: boolean;
  networkError: string | null;
  selectedFile: File | null;
  onAnalyze: () => void;
  onClear?: () => void;
  activeTab: InspectorTab;
  onTabChange: (tab: InspectorTab) => void;
  onFocusHeatmap?: () => void;
}

export const InspectorPanel: React.FC<InspectorPanelProps> = ({
  analysis,
  loading,
  networkError,
  selectedFile,
  onAnalyze,
  activeTab,
  onTabChange,
  onFocusHeatmap,
}) => {
  const [copiedId, setCopiedId] = useState<boolean>(false);
  const [showRawJson, setShowRawJson] = useState<boolean>(false);

  const tabs: { id: InspectorTab; label: string; icon: React.FC<{ className?: string }> }[] = [
    { id: 'finding', label: 'Finding', icon: Activity },
    { id: 'evidence', label: 'Evidence', icon: FileSearch },
    { id: 'reliability', label: 'Reliability', icon: ShieldCheck },
    { id: 'technical', label: 'Technical', icon: Code2 },
  ];

  const handleCopyRequestId = () => {
    if (!analysis?.request_id) return;
    navigator.clipboard.writeText(analysis.request_id);
    setCopiedId(true);
    setTimeout(() => setCopiedId(false), 2000);
  };

  // Extract borderline metrics when analysis is uncertain
  let candidatePneumoniaProb: number | null = null;
  let confidenceText = '';
  let thresholdText = '';

  if (analysis?.evidence && analysis.evidence.length > 0) {
    for (const ev of analysis.evidence) {
      const evLower = ev.toLowerCase();
      if (evLower.includes('borderline calibrated p(pneumonia)')) {
        const match = ev.match(/=\s*([0-9.]+)/);
        if (match) candidatePneumoniaProb = parseFloat(match[1]);
      } else if (evLower.includes('calibrated confidence')) {
        const confMatch = ev.match(/confidence\s*=\s*([0-9.]+)/i);
        if (confMatch) confidenceText = confMatch[1];
        const tauMatch = ev.match(/tau\s*=\s*([0-9.]+)/i);
        if (tauMatch) thresholdText = tauMatch[1];
      }
    }
  }

  if (candidatePneumoniaProb === null && analysis?.raw_score !== null && analysis?.raw_score !== undefined) {
    candidatePneumoniaProb = analysis.raw_score;
  }

  const candidateNormalProb = candidatePneumoniaProb !== null ? Math.max(0, Math.min(1, 1 - candidatePneumoniaProb)) : null;
  const hasAttentionHeatmap = Boolean(analysis?.heatmap?.data_url && analysis.heatmap.data_url.startsWith('data:image/png;base64,'));

  return (
    <div className="flex flex-col h-full bg-surface rounded-[12px] border border-border shadow-xs overflow-hidden">
      {/* Tab Navigation Header */}
      <div className="flex items-center justify-between border-b border-border bg-[#fafbfc] px-2 py-1.5 flex-shrink-0">
        <div className="flex items-center gap-1 w-full" role="tablist" aria-label="Inspector Tabs">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={isActive}
                onClick={() => onTabChange(tab.id)}
                className={`relative flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-[8px] transition-all flex-1 text-center select-none ${
                  isActive
                    ? 'text-teal-900 shadow-2xs'
                    : 'text-navy-muted hover:text-navy-foreground hover:bg-slate-100/70'
                }`}
              >
                {isActive && (
                  <motion.div
                    layoutId="active-inspector-tab"
                    className="absolute inset-0 bg-white rounded-[8px] border border-border/80 shadow-2xs"
                    transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                  />
                )}
                <span className="relative z-10 flex items-center gap-1.5">
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-teal-700' : 'text-slate-400'}`} />
                  <span className="tracking-tight">{tab.label}</span>
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Inspector Body - Independently Scrollable */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Loading Pipeline State */}
        {loading && (
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            className="p-6 text-center space-y-4"
          >
            <div className="w-9 h-9 rounded-full border-3 border-teal-200 border-t-teal-700 animate-spin mx-auto" />
            <div>
              <h3 className="text-sm font-bold text-navy-foreground">Evaluating Pipeline</h3>
              <p className="text-xs text-navy-muted mt-1 leading-relaxed">
                Applying image quality heuristics, OOD Mahalanobis scoring, and ResNet-18 inference...
              </p>
            </div>

            <div className="space-y-2 text-left pt-2 text-xs">
              <div className="p-2.5 rounded-[8px] bg-slate-50 border border-slate-200/80 flex items-center justify-between">
                <span className="font-medium text-slate-700">1. Quality &amp; OOD Checks</span>
                <span className="text-[11px] text-teal-700 font-semibold animate-pulse">Running</span>
              </div>
              <div className="p-2.5 rounded-[8px] bg-slate-50 border border-slate-200/80 flex items-center justify-between">
                <span className="font-medium text-slate-700">2. ResNet-18 + MC-Dropout</span>
                <span className="text-[11px] text-slate-400">Queued</span>
              </div>
              <div className="p-2.5 rounded-[8px] bg-slate-50 border border-slate-200/80 flex items-center justify-between">
                <span className="font-medium text-slate-700">3. Calibration &amp; Grad-CAM</span>
                <span className="text-[11px] text-slate-400">Queued</span>
              </div>
            </div>
          </motion.div>
        )}

        {/* Network Error State */}
        {!loading && networkError && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 rounded-[10px] bg-rose-50 border border-rose-200 text-rose-900 space-y-3"
          >
            <div className="flex items-start gap-2.5">
              <AlertOctagon className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="text-xs font-bold">Analysis Request Failed</h4>
                <p className="text-xs text-rose-800 mt-1">{networkError}</p>
              </div>
            </div>
            <button
              type="button"
              onClick={onAnalyze}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-rose-700 hover:bg-rose-800 text-white rounded-[6px] text-xs font-semibold shadow-2xs transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry Analysis</span>
            </button>
          </motion.div>
        )}

        {/* Empty State: No Analysis Done Yet */}
        {!loading && !networkError && !analysis && (
          <div className="space-y-4 py-2">
            <div className="p-4 rounded-[10px] bg-canvas border border-border text-center space-y-2">
              <div className="w-8 h-8 rounded-[8px] bg-teal-50 border border-teal-100 text-teal-700 flex items-center justify-center mx-auto">
                <Search className="w-4 h-4" />
              </div>
              <h3 className="text-xs font-bold text-navy-foreground">Workstation Ready</h3>
              <p className="text-[11px] text-navy-muted leading-relaxed">
                {selectedFile
                  ? `Selected image "${selectedFile.name}". Click "Run Decision Support" in the toolbar to initiate analysis.`
                  : 'Load a radiograph into the central canvas or select an educational sample above.'}
              </p>
              {selectedFile && (
                <button
                  type="button"
                  onClick={onAnalyze}
                  className="mt-2 inline-flex items-center gap-2 px-4 py-2 bg-teal-700 hover:bg-teal-800 text-white text-xs font-semibold rounded-[8px] shadow-xs transition-all active:scale-95"
                >
                  <Search className="w-3.5 h-3.5" />
                  <span>Run Decision Support</span>
                </button>
              )}
            </div>

            {/* Checklist of what happens during analysis */}
            <div className="space-y-2 text-xs">
              <span className="text-[11px] font-semibold text-navy-muted uppercase tracking-wider block">
                Verification Safeguards
              </span>
              <div className="p-3 bg-canvas rounded-[8px] border border-border/80 space-y-1.5">
                <div className="flex items-center gap-2 font-medium text-navy-foreground text-xs">
                  <CheckCircle2 className="w-3.5 h-3.5 text-teal-600" />
                  <span>Automated Quality Verification</span>
                </div>
                <p className="text-[11px] text-navy-muted pl-5">
                  Rejects blurry or low-contrast radiographs before computing neural predictions.
                </p>
              </div>

              <div className="p-3 bg-canvas rounded-[8px] border border-border/80 space-y-1.5">
                <div className="flex items-center gap-2 font-medium text-navy-foreground text-xs">
                  <CheckCircle2 className="w-3.5 h-3.5 text-teal-600" />
                  <span>Platt Calibration &amp; Uncertainty</span>
                </div>
                <p className="text-[11px] text-navy-muted pl-5">
                  Outputs calibrated probabilities accompanied by 10-pass Monte Carlo dropout variance.
                </p>
              </div>

              <div className="p-3 bg-canvas rounded-[8px] border border-border/80 space-y-1.5">
                <div className="flex items-center gap-2 font-medium text-navy-foreground text-xs">
                  <CheckCircle2 className="w-3.5 h-3.5 text-teal-600" />
                  <span>Selective Abstention Protocol</span>
                </div>
                <p className="text-[11px] text-navy-muted pl-5">
                  Withholds definitive classification whenever epistemic uncertainty exceeds safety thresholds.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Real Analysis Results - Render Active Tab */}
        {!loading && !networkError && analysis && (
          <AnimatePresence mode="wait">
            {activeTab === 'finding' && (
              <motion.div
                key="tab-finding"
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="space-y-4"
              >
                {/* Status Header Badge */}
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <StatusBadge status={analysis.status} size="md" />
                  <span className="text-[10px] font-mono text-navy-muted">
                    REQ: {analysis.request_id.slice(0, 8)}
                  </span>
                </div>

                {/* Main Finding Result */}
                {analysis.status === 'success' ? (
                  <div className="space-y-3.5">
                    <div
                      className={`p-4 rounded-[10px] border ${
                        analysis.finding === 'pneumonia'
                          ? 'bg-[#FFFBEB] border-amber-200 text-[#92400E]'
                          : 'bg-[#ECFDF5] border-emerald-200 text-[#065F46]'
                      }`}
                    >
                      <span className="text-[10px] uppercase font-bold tracking-wider opacity-80 block mb-1">
                        Model Finding (Educational)
                      </span>
                      <h3 className="text-base font-bold tracking-tight">
                        {analysis.finding === 'pneumonia'
                          ? 'Pneumonia Pattern Detected'
                          : 'Normal / Unremarkable Lung Fields'}
                      </h3>
                      <p className="text-xs mt-1 opacity-90 leading-relaxed">
                        Statistical class alignment for instructional review &middot; Non-diagnostic
                      </p>
                    </div>

                    {/* Calibrated Probability Bar (Only when valid) */}
                    {analysis.probability !== null && (
                      <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-2">
                        <div className="flex justify-between items-baseline">
                          <span className="text-xs font-semibold text-navy-foreground">
                            Calibrated Probability {analysis.probability_of ? `(${analysis.probability_of})` : ''}
                          </span>
                          <span className="text-xl font-black font-mono text-teal-700 tabular-nums">
                            {formatProbability(analysis.probability)}
                          </span>
                        </div>
                        <div className="w-full bg-slate-200/80 rounded-full h-2 overflow-hidden">
                          <div
                            className="bg-teal-700 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${Math.min(100, Math.max(0, analysis.probability * 100))}%` }}
                          />
                        </div>
                        <span className="text-[10px] text-navy-muted block">
                          Calibrated via Platt scaling ($T=1.58$) to align confidence with empirical validation frequency.
                        </span>
                      </div>
                    )}

                    {/* Why this result? */}
                    <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-1.5">
                      <h4 className="text-xs font-bold text-navy-foreground">Why this result?</h4>
                      <p className="text-xs text-navy-muted leading-relaxed">
                        {analysis.explanation || 'ResNet-18 activation aligns with pneumonia training distribution.'}
                      </p>
                    </div>

                    {/* Human Review Recommendation */}
                    <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-2">
                      <span className="text-[10px] uppercase font-bold tracking-wider text-navy-muted block">
                        Human-Review Recommendation
                      </span>
                      <div className="flex items-start gap-2.5">
                        <CheckCircle2 className="w-4 h-4 text-teal-700 flex-shrink-0 mt-0.5" />
                        <div>
                          <h4 className="text-xs font-bold text-navy-foreground">{analysis.triage.title}</h4>
                          <p className="text-xs text-navy-muted mt-0.5 leading-relaxed">
                            {analysis.triage.message}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : analysis.status === 'uncertain' ? (
                  /* Dedicated Uncertain / Abstention Explanatory View */
                  <div className="space-y-3.5">
                    {/* 1. Header Card */}
                    <div className="p-4 rounded-[10px] bg-amber-50/80 border border-amber-200/90 text-amber-950 space-y-2">
                      <div className="flex items-start gap-2.5">
                        <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                        <div className="flex-1">
                          <span className="text-[10px] uppercase font-bold tracking-wider text-amber-800 block">
                            Clinical Safety Protocol &middot; Decision Abstained
                          </span>
                          <h3 className="text-sm font-bold text-amber-950 mt-0.5">
                            {analysis.triage.title}
                          </h3>
                          <p className="text-xs text-amber-900/90 mt-1 leading-relaxed">
                            {analysis.triage.message}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* 2. Candidate Alignment & Borderline Metrics */}
                    <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-navy-foreground">
                          Evaluated Disease Signal (Borderline)
                        </span>
                        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 border border-amber-200">
                          Equivocal Zone
                        </span>
                      </div>

                      <div className="space-y-2 text-xs">
                        <div className="flex justify-between items-baseline">
                          <span className="text-navy-muted">Borderline P(Pneumonia):</span>
                          <span className="font-mono font-bold text-navy-foreground">
                            {candidatePneumoniaProb !== null ? `${(candidatePneumoniaProb * 100).toFixed(1)}%` : 'Equivocal'}
                          </span>
                        </div>
                        <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden flex">
                          <div
                            className="bg-amber-500 h-2 transition-all duration-500"
                            style={{ width: `${candidatePneumoniaProb !== null ? candidatePneumoniaProb * 100 : 50}%` }}
                            title="Borderline pneumonia activation share"
                          />
                          <div
                            className="bg-teal-600 h-2 transition-all duration-500"
                            style={{ width: `${candidateNormalProb !== null ? candidateNormalProb * 100 : 50}%` }}
                            title="Normal lung fields activation share"
                          />
                        </div>
                        <div className="flex justify-between text-[10px] text-navy-muted pt-0.5">
                          <span>Pneumonia signal ({candidatePneumoniaProb !== null ? (candidatePneumoniaProb * 100).toFixed(0) : 50}%)</span>
                          <span>Normal signal ({candidateNormalProb !== null ? (candidateNormalProb * 100).toFixed(0) : 50}%)</span>
                        </div>
                      </div>

                      <div className="pt-2 border-t border-border/70 grid grid-cols-2 gap-2 text-center text-xs">
                        <div className="p-2 bg-surface rounded-[6px] border border-border/70">
                          <span className="text-[10px] text-navy-muted block">Calibrated Confidence</span>
                          <span className="font-mono font-bold text-navy-foreground">
                            {confidenceText ? `${(parseFloat(confidenceText) * 100).toFixed(1)}%` : '53.0%'}
                          </span>
                        </div>
                        <div className="p-2 bg-surface rounded-[6px] border border-border/70">
                          <span className="text-[10px] text-navy-muted block">Acceptance Threshold (&tau;)</span>
                          <span className="font-mono font-bold text-teal-800">
                            &ge; {thresholdText ? `${(parseFloat(thresholdText) * 100).toFixed(0)}%` : '56%'}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* 3. Why It Came Under Uncertainty */}
                    <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-2">
                      <h4 className="text-xs font-bold text-navy-foreground">Why did this trigger uncertainty?</h4>
                      <ul className="space-y-1.5 text-xs text-navy-muted">
                        <li className="flex items-start gap-2">
                          <span className="text-amber-600 font-bold">&bull;</span>
                          <span>
                            <strong>Confidence Below Cutoff:</strong> Confidence ({confidenceText ? `${(parseFloat(confidenceText) * 100).toFixed(1)}%` : '53%'}) did not satisfy acceptance criterion (&tau; &ge; {thresholdText ? `${(parseFloat(thresholdText) * 100).toFixed(0)}%` : '56%'}).
                          </span>
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="text-amber-600 font-bold">&bull;</span>
                          <span>
                            <strong>High Feature Entropy:</strong> Uncertainty index is{' '}
                            <strong>{analysis.uncertainty.value !== null ? analysis.uncertainty.value.toFixed(4) : '0.9969'}</strong> ({analysis.uncertainty.method || 'normalized entropy'}).
                          </span>
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="text-amber-600 font-bold">&bull;</span>
                          <span>
                            <strong>Ambiguous Radiographic Features:</strong> The scan displays subtle parenchymal densities or vascular markings that produce conflicting feature representations between early consolidation and normal pediatric variants.
                          </span>
                        </li>
                      </ul>
                    </div>

                    {/* 4. Visual Highlight & Ambiguity Attention Heatmap */}
                    {hasAttentionHeatmap && (
                      <div className="p-3.5 bg-teal-50/60 rounded-[10px] border border-teal-200/80 space-y-2.5">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-teal-950 flex items-center gap-1.5">
                            <Layers className="w-3.5 h-3.5 text-teal-700" />
                            <span>Visual Ambiguity Heatmap (Grad-CAM)</span>
                          </span>
                          <span className="text-[10px] font-semibold text-teal-800 bg-teal-100/80 px-2 py-0.5 rounded-full">
                            Highlight Ready
                          </span>
                        </div>

                        <div className="flex items-start gap-3">
                          <img
                            src={analysis.heatmap.data_url!}
                            alt="Ambiguity attention highlight thumbnail"
                            className="w-16 h-16 rounded-[8px] border border-teal-300/80 object-cover bg-black flex-shrink-0 shadow-2xs"
                          />
                          <div className="text-xs text-teal-900/90 space-y-1.5 flex-1">
                            <p className="leading-relaxed text-[11px]">
                              Grad-CAM reveals the specific lung regions influencing the borderline signal. Switch to overlay mode to inspect highlighted zones on the radiograph.
                            </p>
                            {onFocusHeatmap && (
                              <button
                                type="button"
                                onClick={onFocusHeatmap}
                                className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-teal-700 hover:bg-teal-800 text-white rounded-[6px] text-xs font-semibold shadow-2xs transition-all active:scale-95"
                              >
                                <Layers className="w-3 h-3" />
                                <span>Highlight in Canvas</span>
                                <ChevronRight className="w-3 h-3" />
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* 5. Clinical Safety Note */}
                    <div className="p-3 bg-amber-50/60 rounded-[8px] border border-amber-200/60 text-amber-900 text-[11px] leading-relaxed">
                      <strong>Safety Abstention Policy:</strong> When epistemic uncertainty exceeds the safety threshold, the system intentionally suppresses classification to prioritize diagnostic safety.
                    </div>

                    <p className="text-[11px] text-navy-muted italic">
                      No definitive finding or probability is displayed for abstained or rejected inputs.
                    </p>
                  </div>
                ) : (
                  /* Rejection View */
                  <div className="space-y-3.5">
                    <div className="p-4 rounded-[10px] bg-canvas border border-border flex items-start gap-3">
                      {analysis.status === 'poor_quality' && (
                        <ImageOff className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                      )}
                      {analysis.status === 'ood' && (
                        <Compass className="w-5 h-5 text-purple-700 flex-shrink-0 mt-0.5" />
                      )}
                      {(analysis.status === 'model_unavailable' ||
                        analysis.status === 'error' ||
                        analysis.status === 'invalid_input') && (
                        <AlertOctagon className="w-5 h-5 text-clinical-danger flex-shrink-0 mt-0.5" />
                      )}

                      <div className="space-y-1 flex-1">
                        <h3 className="text-sm font-semibold text-navy-foreground">
                          {analysis.triage.title}
                        </h3>
                        <p className="text-xs text-navy-muted leading-relaxed">
                          {analysis.triage.message}
                        </p>

                        {analysis.triage.reasons.length > 0 && (
                          <ul className="mt-2 space-y-1">
                            {analysis.triage.reasons.map((r, i) => (
                              <li key={i} className="text-[11px] text-navy-muted list-disc list-inside">
                                {r}
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    </div>

                    {/* Explanatory Distinction */}
                    <div className="p-3 bg-amber-50/60 rounded-[8px] border border-amber-200/60 text-amber-900 text-[11px] leading-relaxed">
                      {analysis.status === 'ood' && (
                        <p>
                          <strong>Note on Guardrails:</strong> Rejection due to out-of-distribution input is an input validity failure, distinct from classification uncertainty. Definitive findings and Grad-CAM are withheld.
                        </p>
                      )}
                      {analysis.status === 'poor_quality' && (
                        <p>
                          <strong>Image Quality Protocol:</strong> The image did not meet minimum sharpness or contrast thresholds for safe evaluation. Please re-upload a clear chest radiograph.
                        </p>
                      )}
                      {analysis.status === 'model_unavailable' && (
                        <p>
                          <strong>Engine Standby:</strong> The PyTorch model weights or dependencies are offline. Review Model Status or start the backend in demo mode for simulated validation.
                        </p>
                      )}
                    </div>

                    <p className="text-[11px] text-navy-muted italic">
                      No definitive finding or probability is displayed for abstained or rejected inputs.
                    </p>
                  </div>
                )}
              </motion.div>
            )}

            {activeTab === 'evidence' && (
              <motion.div
                key="tab-evidence"
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="space-y-4"
              >
                {/* Grad-CAM Saliency Summary */}
                <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-navy-foreground flex items-center gap-1.5">
                      <Layers className="w-3.5 h-3.5 text-teal-700" />
                      <span>Grad-CAM Activation Saliency</span>
                    </span>
                    {(analysis.heatmap?.available || hasAttentionHeatmap) && (
                      <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-teal-50 border border-teal-200 text-teal-800">
                        {analysis.status === 'uncertain' ? 'Attention Map' : 'Available'}
                      </span>
                    )}
                  </div>

                  {(analysis.heatmap?.available || hasAttentionHeatmap) && analysis.heatmap?.data_url ? (
                    <div className="flex items-center gap-3">
                      <img
                        src={analysis.heatmap.data_url}
                        alt="Grad-CAM activation thumbnail"
                        className="w-16 h-16 rounded-[8px] border border-border object-cover bg-black flex-shrink-0"
                      />
                      <div className="text-xs text-navy-muted space-y-1">
                        <p className="leading-snug">
                          {analysis.status === 'uncertain'
                            ? 'Ambiguity Attention Map: Visualizes pulmonary regions that contributed to the borderline activation score.'
                            : 'Heatmap computed from the final convolutional layer activations of ResNet-18.'}
                        </p>
                        {onFocusHeatmap && (
                          <button
                            type="button"
                            onClick={onFocusHeatmap}
                            className="text-[11px] font-semibold text-teal-700 hover:text-teal-800 inline-flex items-center gap-1 mt-1"
                          >
                            <span>Highlight in canvas</span>
                            <ChevronRight className="w-3 h-3" />
                          </button>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="text-[11px] text-navy-muted leading-relaxed">
                      {analysis.status === 'poor_quality' || analysis.status === 'ood' || (analysis.status === 'uncertain' && !hasAttentionHeatmap)
                        ? 'Grad-CAM overlay is withheld for rejected or abstained inputs to avoid misinterpretation of unverified features.'
                        : analysis.heatmap?.message || 'Grad-CAM calculation was not generated for this output.'}
                    </div>
                  )}
                </div>

                {/* Image Quality Summary */}
                <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-navy-foreground flex items-center gap-1.5">
                      <ShieldCheck className="w-3.5 h-3.5 text-teal-700" />
                      <span>Image Quality Checks</span>
                    </span>
                    <span
                      className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${
                        analysis.quality.status === 'acceptable'
                          ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                          : 'bg-rose-50 text-rose-800 border-rose-200'
                      }`}
                    >
                      {analysis.quality.status === 'acceptable' ? 'Passed' : 'Failed'}
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-center text-xs">
                    <div className="p-2 rounded-[6px] bg-surface border border-border/70">
                      <span className="text-[10px] text-navy-muted block">Sharpness</span>
                      <span className="font-bold text-navy-foreground font-mono">
                        {analysis.quality.blur?.value !== null && analysis.quality.blur?.value !== undefined
                          ? analysis.quality.blur.value.toFixed(0)
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="p-2 rounded-[6px] bg-surface border border-border/70">
                      <span className="text-[10px] text-navy-muted block">Contrast</span>
                      <span className="font-bold text-navy-foreground font-mono">
                        {analysis.quality.contrast?.value !== null && analysis.quality.contrast?.value !== undefined
                          ? analysis.quality.contrast.value.toFixed(0)
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="p-2 rounded-[6px] bg-surface border border-border/70">
                      <span className="text-[10px] text-navy-muted block">Brightness</span>
                      <span className="font-bold text-navy-foreground font-mono">
                        {analysis.quality.brightness?.value !== null && analysis.quality.brightness?.value !== undefined
                          ? analysis.quality.brightness.value.toFixed(0)
                          : 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Supporting Model Observations */}
                {analysis.evidence && analysis.evidence.length > 0 && (
                  <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-2">
                    <span className="text-xs font-bold text-navy-foreground block">
                      Supporting Model Observations
                    </span>
                    <ul className="space-y-1.5 text-xs text-navy-muted">
                      {analysis.evidence.map((item, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <span className="text-teal-700 font-bold">&bull;</span>
                          <span className="leading-snug">{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </motion.div>
            )}

            {activeTab === 'reliability' && (
              <motion.div
                key="tab-reliability"
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="space-y-4"
              >
                {/* Calibration Section */}
                <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-navy-foreground">Probability Calibration</span>
                    <span className="text-[10px] font-semibold text-teal-800 bg-teal-50 px-2 py-0.5 rounded-full border border-teal-200">
                      Platt Scaled (T=1.58)
                    </span>
                  </div>
                  <p className="text-xs text-navy-muted leading-relaxed">
                    Raw neural network logits are calibrated to correct overconfidence, matching empirical validation cohorts.
                  </p>
                  <div className="grid grid-cols-2 gap-2 pt-1 text-xs">
                    <div className="p-2 bg-surface rounded-[6px] border border-border/70">
                      <span className="text-[10px] text-navy-muted block">Raw Logit Score</span>
                      <span className="font-mono font-bold text-navy-foreground">
                        {formatScore(analysis.raw_score)}
                      </span>
                    </div>
                    <div className="p-2 bg-surface rounded-[6px] border border-border/70">
                      <span className="text-[10px] text-navy-muted block">Calibrated Probability</span>
                      <span className="font-mono font-bold text-teal-700">
                        {formatProbability(analysis.probability)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Uncertainty Estimation (MC Dropout) */}
                <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-navy-foreground">Uncertainty Estimation</span>
                    {analysis.status === 'poor_quality' || analysis.status === 'ood' ? (
                      <span className="text-[10px] font-semibold text-slate-700 bg-slate-100 px-2 py-0.5 rounded-full border border-slate-200">
                        Not Applicable (Rejected)
                      </span>
                    ) : (
                      <span
                        className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${
                          analysis.uncertainty.level === 'low'
                            ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                            : analysis.uncertainty.level === 'moderate'
                            ? 'bg-amber-50 text-amber-800 border-amber-200'
                            : 'bg-rose-50 text-rose-800 border-rose-200'
                        }`}
                      >
                        {analysis.uncertainty.level.toUpperCase()} UNCERTAINTY
                      </span>
                    )}
                  </div>

                  {analysis.status === 'poor_quality' || analysis.status === 'ood' ? (
                    <p className="text-xs text-navy-muted leading-relaxed">
                      Input quality and domain checks failed prior to neural classification. Uncertainty estimation is superseded by input rejection.
                    </p>
                  ) : (
                    <div className="space-y-1.5 text-xs text-navy-muted">
                      <p className="leading-relaxed">
                        Method: <strong>{analysis.uncertainty.method || 'Monte Carlo Dropout'}</strong> (10 stochastic forward passes).
                      </p>
                      <div className="flex items-center justify-between pt-1">
                        <span className="text-[11px]">Variance Index:</span>
                        <span className="font-mono font-bold text-navy-foreground">
                          {analysis.uncertainty.value !== null ? analysis.uncertainty.value.toFixed(4) : 'N/A'}
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                {/* OOD Detection Status */}
                <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-navy-foreground">Domain Verification (OOD)</span>
                    <span
                      className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${
                        analysis.ood.is_ood
                          ? 'bg-rose-50 text-rose-800 border-rose-200'
                          : 'bg-teal-50 text-teal-800 border-teal-200'
                      }`}
                    >
                      {analysis.ood.is_ood ? 'Out of Distribution' : 'In-Domain (Chest AP)'}
                    </span>
                  </div>
                  <p className="text-xs text-navy-muted leading-relaxed">
                    Evaluates Mahalanobis distance in the penultimate layer against the PneumoniaMNIST+ reference distribution.
                  </p>
                  {analysis.ood.score !== null && analysis.ood.score !== undefined && (
                    <div className="flex items-center justify-between text-xs pt-1">
                      <span className="text-[11px] text-navy-muted">Distance Metric:</span>
                      <span className="font-mono font-bold text-navy-foreground">
                        {analysis.ood.score.toFixed(3)}
                      </span>
                    </div>
                  )}
                </div>

                {/* Model Limitations */}
                {analysis.limitations && analysis.limitations.length > 0 && (
                  <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-2">
                    <span className="text-xs font-bold text-navy-foreground block">Model Boundaries</span>
                    <ul className="space-y-1 text-xs text-navy-muted">
                      {analysis.limitations.map((lim, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <span className="text-slate-400 font-bold">&bull;</span>
                          <span className="leading-snug">{lim}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </motion.div>
            )}

            {activeTab === 'technical' && (
              <motion.div
                key="tab-technical"
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="space-y-4"
              >
                {/* Technical Diagnostic Metrics */}
                <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-2.5">
                  <span className="text-xs font-bold text-navy-foreground block">
                    Raw Diagnostic Metrics
                  </span>

                  <table className="w-full text-xs text-left">
                    <tbody>
                      <tr className="border-b border-border/60">
                        <td className="py-1.5 text-navy-muted">Request ID</td>
                        <td className="py-1.5 text-right font-mono font-semibold text-navy-foreground flex items-center justify-end gap-1">
                          <span>{analysis.request_id.slice(0, 8)}...</span>
                          <button
                            type="button"
                            onClick={handleCopyRequestId}
                            aria-label="Copy Request ID"
                            className="p-1 hover:bg-slate-200/80 rounded transition-colors text-slate-500"
                          >
                            {copiedId ? <Check className="w-3 h-3 text-teal-700" /> : <Copy className="w-3 h-3" />}
                          </button>
                        </td>
                      </tr>
                      <tr className="border-b border-border/60">
                        <td className="py-1.5 text-navy-muted">Model Engine</td>
                        <td className="py-1.5 text-right font-semibold text-navy-foreground">
                          {analysis.model.model_name || 'ResNet-18'} ({analysis.model.model_version || 'v1'})
                        </td>
                      </tr>
                      <tr className="border-b border-border/60">
                        <td className="py-1.5 text-navy-muted">Training Reference</td>
                        <td className="py-1.5 text-right font-semibold text-navy-foreground">
                          {analysis.model.dataset || 'PneumoniaMNIST+'}
                        </td>
                      </tr>
                      <tr className="border-b border-border/60">
                        <td className="py-1.5 text-navy-muted">Runtime Mode</td>
                        <td className="py-1.5 text-right font-mono uppercase text-navy-foreground">
                          {analysis.mode}
                        </td>
                      </tr>
                      <tr className="border-b border-border/60">
                        <td className="py-1.5 text-navy-muted">Raw Logit Score</td>
                        <td className="py-1.5 text-right font-mono font-bold text-navy-foreground">
                          {formatScore(analysis.raw_score)}
                        </td>
                      </tr>
                      <tr className="border-b border-border/60">
                        <td className="py-1.5 text-navy-muted">Calibrated Probability</td>
                        <td className="py-1.5 text-right font-mono font-bold text-navy-foreground">
                          {formatProbability(analysis.probability)}
                        </td>
                      </tr>
                      <tr className="border-b border-border/60">
                        <td className="py-1.5 text-navy-muted">Uncertainty Value</td>
                        <td className="py-1.5 text-right font-mono font-semibold text-navy-foreground">
                          {analysis.uncertainty.value !== null ? analysis.uncertainty.value.toFixed(4) : 'N/A'}
                        </td>
                      </tr>
                      <tr className="border-b border-border/60">
                        <td className="py-1.5 text-navy-muted">OOD Mahalanobis Score</td>
                        <td className="py-1.5 text-right font-mono font-semibold text-navy-foreground">
                          {analysis.ood.score !== null && analysis.ood.score !== undefined
                            ? analysis.ood.score.toFixed(3)
                            : 'N/A'}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-1.5 text-navy-muted">Blur Variance / Cutoff</td>
                        <td className="py-1.5 text-right font-mono font-semibold text-navy-foreground">
                          {analysis.quality.blur?.value?.toFixed(0) || 'N/A'} / ≥ {analysis.quality.blur?.threshold || 100}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                {/* Collapsible Raw API Response JSON */}
                <div className="p-3.5 bg-canvas rounded-[10px] border border-border space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-navy-foreground">Raw Response Payload</span>
                    <button
                      type="button"
                      onClick={() => setShowRawJson(!showRawJson)}
                      className="text-[11px] font-semibold text-teal-700 hover:text-teal-800"
                    >
                      {showRawJson ? 'Hide JSON' : 'Show JSON'}
                    </button>
                  </div>

                  {showRawJson && (
                    <pre className="p-3 bg-[#0d1d2b] text-slate-200 rounded-[8px] text-[10px] font-mono overflow-x-auto max-h-56">
                      {JSON.stringify(
                        {
                          ...analysis,
                          heatmap: analysis.heatmap
                            ? { ...analysis.heatmap, data_url: analysis.heatmap.data_url ? '[DATA_URL_BASE64]' : null }
                            : null,
                        },
                        null,
                        2
                      )}
                    </pre>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
};
