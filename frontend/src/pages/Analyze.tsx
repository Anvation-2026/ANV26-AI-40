import React, { useEffect, useRef, useState } from 'react';
import {
  Loader2,
  Search,
  Sliders,
  Database,
  FileImage,
} from 'lucide-react';
import { ImageUploader } from '../components/ImageUploader';
import { XrayViewer } from '../components/XrayViewer';
import { FindingCard } from '../components/FindingCard';
import { ConfidencePanel } from '../components/ConfidencePanel';
import { QualityPanel } from '../components/QualityPanel';
import { EvidencePanel } from '../components/EvidencePanel';
import { EscalationPanel } from '../components/EscalationPanel';
import { DemoModeBanner } from '../components/DemoModeBanner';
import { ModelStatusCard } from '../components/ModelStatusCard';
import { ErrorState } from '../components/ErrorState';
import { useToast } from '../components/Toast';
import { getModelStatus, predictImage, NetworkError } from '../services/api';
import { loadSampleAsset, SAMPLE_PRESETS } from '../services/sampleAssets';
import { AnalysisResponse, ModelStatusResponse } from '../types/analysis';

export const Analyze: React.FC = () => {
  const { showToast } = useToast();
  const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(null);
  const [demoScenario, setDemoScenario] = useState<string>('uncertain');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [networkError, setNetworkError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const timeoutIdRef = useRef<number | null>(null);

  useEffect(() => {
    let mounted = true;
    getModelStatus()
      .then((data) => {
        if (mounted) setModelStatus(data);
      })
      .catch(() => {});

    return () => {
      mounted = false;
      if (abortControllerRef.current) abortControllerRef.current.abort();
      if (timeoutIdRef.current) clearTimeout(timeoutIdRef.current);
    };
  }, []);

  const handleFileSelected = (file: File) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setAnalysisResult(null);
    setNetworkError(null);
    showToast(`Loaded ${file.name}`, 'info');
  };

  const handleClear = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (timeoutIdRef.current) {
      clearTimeout(timeoutIdRef.current);
      timeoutIdRef.current = null;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setAnalysisResult(null);
    setNetworkError(null);
    setLoading(false);
  };

  const loadSamplePreset = async (url: string, name: string) => {
    try {
      const file = await loadSampleAsset(url, name);
      handleFileSelected(file);
    } catch {
      showToast('Could not load sample asset', 'error');
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile || loading) return;

    setLoading(true);
    setNetworkError(null);
    setAnalysisResult(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    timeoutIdRef.current = window.setTimeout(() => {
      controller.abort();
      setNetworkError('Analysis request timed out after 60 seconds.');
      showToast('Request timed out', 'error');
      setLoading(false);
    }, 60000);

    try {
      const scenarioParam = modelStatus?.mode === 'demo' ? demoScenario : undefined;
      const res = await predictImage(selectedFile, controller.signal, scenarioParam);
      setAnalysisResult(res);

      if (res.status === 'success') {
        showToast('Decision support evaluation complete', 'success');
      } else if (res.status === 'uncertain') {
        showToast('Decision abstained due to uncertainty', 'info');
      } else {
        showToast(`Status: ${res.status}`, 'info');
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        // Ignored
      } else if (err instanceof NetworkError) {
        setNetworkError(err.message);
        showToast(err.message, 'error');
      } else {
        setNetworkError('A connection error occurred during analysis.');
        showToast('Connection error', 'error');
      }
    } finally {
      if (timeoutIdRef.current) {
        clearTimeout(timeoutIdRef.current);
        timeoutIdRef.current = null;
      }
      setLoading(false);
      abortControllerRef.current = null;
    }
  };

  const isDemo = modelStatus?.mode === 'demo';

  return (
    <div className="space-y-4 max-w-7xl mx-auto">
      {/* Compact Operational Bar */}
      <div className="bg-surface rounded-[12px] border border-border px-4 py-2.5 shadow-xs flex flex-wrap items-center justify-between gap-3 text-xs">
        <div>
          <h2 className="text-sm font-bold text-navy-foreground tracking-tight">
            Decision Support & Triage Workspace
          </h2>
          <span className="text-[11px] text-navy-muted">
            Independent Image Inspection & Grad-CAM Evidence
          </span>
        </div>

        {/* Quick Sample Presets */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[11px] font-medium text-navy-muted mr-1 hidden sm:inline">
            Load Educational Sample:
          </span>
          {SAMPLE_PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              onClick={() => loadSamplePreset(preset.assetUrl, preset.filename)}
              disabled={loading}
              className={`px-2.5 py-1 text-[11px] font-medium border rounded-[6px] transition-colors ${
                preset.category === 'uncertain'
                  ? 'text-amber-800 bg-amber-50 hover:bg-amber-100 border-amber-200'
                  : preset.category === 'normal' || preset.category === 'pneumonia'
                  ? 'text-navy-foreground bg-canvas hover:bg-slate-200/80 border-border'
                  : 'text-navy-muted hover:text-navy-foreground bg-canvas hover:bg-slate-200/80 border-border'
              }`}
              title={preset.description}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* Demo Mode Banner (only when in demo mode) */}
      {isDemo && (
        <DemoModeBanner
          scenario={demoScenario}
          onScenarioChange={setDemoScenario}
        />
      )}

      {/* Main Workspace: 60% Interactive Viewer / 40% Results Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Column (Viewer & Upload) */}
        <div className="lg:col-span-7 space-y-3">
          {!selectedFile ? (
            <div className="bg-surface rounded-[12px] border border-border p-6 shadow-xs">
              <div className="mb-3.5">
                <h3 className="text-sm font-bold text-navy-foreground tracking-tight">
                  Radiograph Upload
                </h3>
                <p className="text-xs text-navy-muted">
                  Drag and drop a public or de-identified educational radiograph (PNG, JPEG, WEBP &le; 10 MB).
                </p>
              </div>
              <ImageUploader onFileSelected={handleFileSelected} disabled={loading} />
            </div>
          ) : (
            <div className="space-y-3">
              {/* Interactive Medical Image Viewer */}
              <XrayViewer
                originalUrl={previewUrl}
                heatmap={analysisResult?.heatmap || null}
                status={analysisResult?.status}
              />

              {/* Action Toolbar */}
              <div className="bg-surface rounded-[12px] border border-border p-3 shadow-xs flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2 truncate text-xs text-navy-muted">
                  <FileImage className="w-4 h-4 text-teal-700 flex-shrink-0" />
                  <span className="font-semibold text-navy-foreground truncate max-w-[200px]">
                    {selectedFile.name}
                  </span>
                  <span>&middot;</span>
                  <span className="font-mono tabular-nums">
                    {(selectedFile.size / 1024).toFixed(1)} KB
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleAnalyze}
                    disabled={loading}
                    className="inline-flex items-center justify-center gap-2 px-4 py-2 bg-teal-700 hover:bg-teal-800 disabled:bg-slate-300 text-white font-semibold text-xs rounded-[8px] shadow-xs transition-colors focus:outline-none focus:ring-2 focus:ring-teal-600"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>Evaluating...</span>
                      </>
                    ) : (
                      <>
                        <Search className="w-3.5 h-3.5" />
                        <span>{analysisResult ? 'Re-evaluate' : 'Run Decision Support'}</span>
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={handleClear}
                    disabled={loading}
                    className="px-3 py-2 bg-canvas hover:bg-slate-200/70 text-navy-muted hover:text-navy-foreground font-medium text-xs rounded-[8px] border border-border transition-colors focus:outline-none focus:ring-2 focus:ring-slate-300"
                  >
                    Clear Image
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Model Status Bar */}
          <ModelStatusCard compact={false} onStatusLoaded={setModelStatus} />
        </div>

        {/* Right Column: Diagnostic & Triage Guidance */}
        <div className="lg:col-span-5 space-y-3" aria-live="polite">
          {loading && (
            <div className="bg-surface rounded-[12px] border border-border p-8 text-center shadow-xs space-y-3">
              <div className="w-8 h-8 rounded-full border-3 border-teal-200 border-t-teal-700 animate-spin mx-auto"></div>
              <div>
                <h3 className="text-sm font-bold text-navy-foreground">
                  Evaluating Radiograph
                </h3>
                <p className="text-xs text-navy-muted mt-1 max-w-xs mx-auto leading-relaxed">
                  Executing quality verification, out-of-distribution distance scoring, calibrated probability, and Grad-CAM activations...
                </p>
              </div>

              <div className="space-y-2 pt-2">
                <div className="h-12 bg-slate-100 rounded-[8px] animate-pulse"></div>
                <div className="h-20 bg-slate-100 rounded-[8px] animate-pulse"></div>
              </div>
            </div>
          )}

          {networkError && (
            <ErrorState
              message={networkError}
              onRetry={handleAnalyze}
            />
          )}

          {!loading && !analysisResult && !networkError && (
            <div className="bg-surface rounded-[12px] border border-dashed border-border p-8 text-center shadow-xs">
              <Sliders className="w-7 h-7 mx-auto text-navy-muted opacity-40 mb-2" />
              <h3 className="text-sm font-bold text-navy-foreground">
                Awaiting Analysis
              </h3>
              <p className="text-xs text-navy-muted mt-1 max-w-xs mx-auto leading-relaxed">
                Upload a radiograph or pick an educational preset to view triage results, probability calibration, and Grad-CAM attention maps.
              </p>
            </div>
          )}

          {!loading && analysisResult && (
            <div className="space-y-3">
              {/* Finding Card */}
              <FindingCard analysis={analysisResult} />

              {/* Confidence & Uncertainty */}
              <ConfidencePanel analysis={analysisResult} />

              {/* Quality & Domain Guardrails */}
              <QualityPanel
                quality={analysisResult.quality}
                ood={analysisResult.ood}
              />

              {/* Model Evidence & Explanations */}
              <EvidencePanel
                explanation={analysisResult.explanation}
                evidence={analysisResult.evidence}
                limitations={analysisResult.limitations}
              />

              {/* Educational Escalation Protocol */}
              <EscalationPanel triage={analysisResult.triage} />

              {/* Provenance Footer */}
              <div className="p-3 bg-canvas rounded-[10px] border border-border text-[11px] text-navy-muted flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2.5 truncate">
                  <Database className="w-3.5 h-3.5 text-teal-700 flex-shrink-0" />
                  <span className="truncate">
                    {analysisResult.model.model_name || 'ResNet-18'} ({analysisResult.model.model_version || 'v1'})
                  </span>
                  <span>&middot;</span>
                  <span>{analysisResult.model.dataset || 'PneumoniaMNIST+'}</span>
                </div>
                <span className="font-mono text-[10px] text-navy-muted">
                  ID: {analysisResult.request_id.slice(0, 8)}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
