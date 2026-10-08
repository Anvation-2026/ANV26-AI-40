import React, { useEffect, useRef, useState } from 'react';
import {
  Loader2,
  RefreshCw,
  Search,
  Sliders,
  Database,
  Tag,
  Hash,
} from 'lucide-react';
import { ImageUploader } from '../components/ImageUploader';
import { ImagePreview } from '../components/ImagePreview';
import { FindingCard } from '../components/FindingCard';
import { ConfidencePanel } from '../components/ConfidencePanel';
import { QualityPanel } from '../components/QualityPanel';
import { HeatmapViewer } from '../components/HeatmapViewer';
import { EvidencePanel } from '../components/EvidencePanel';
import { EscalationPanel } from '../components/EscalationPanel';
import { DemoModeBanner } from '../components/DemoModeBanner';
import { ErrorState } from '../components/ErrorState';
import { getModelStatus, predictImage, NetworkError } from '../services/api';
import { AnalysisResponse, ModelStatusResponse } from '../types/analysis';

export const Analyze: React.FC = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(null);
  const [demoScenario, setDemoScenario] = useState<string>('uncertain');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [networkError, setNetworkError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const timeoutIdRef = useRef<number | null>(null);

  // Check model status on mount
  useEffect(() => {
    let mounted = true;
    getModelStatus()
      .then((data) => {
        if (mounted) setModelStatus(data);
      })
      .catch(() => {
        // Handled silently or on submit
      });

    return () => {
      mounted = false;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (timeoutIdRef.current) {
        clearTimeout(timeoutIdRef.current);
      }
    };
  }, []);

  const handleFileSelected = (file: File) => {
    // Clean up previous object URL if any
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setAnalysisResult(null);
    setNetworkError(null);
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
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setAnalysisResult(null);
    setNetworkError(null);
    setLoading(false);
  };

  const handleAnalyze = async () => {
    if (!selectedFile || loading) return;

    setLoading(true);
    setNetworkError(null);
    setAnalysisResult(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    // 60-second client-side safety timeout
    timeoutIdRef.current = window.setTimeout(() => {
      controller.abort();
      setNetworkError('Analysis request timed out after 60 seconds. Please check the backend connection.');
      setLoading(false);
    }, 60000);

    try {
      const scenarioParam = modelStatus?.mode === 'demo' ? demoScenario : undefined;
      const res = await predictImage(selectedFile, controller.signal, scenarioParam);
      setAnalysisResult(res);
    } catch (err: any) {
      if (err.name === 'AbortError') {
        // Ignored or handled if triggered by timeout
      } else if (err instanceof NetworkError) {
        setNetworkError(err.message);
      } else {
        setNetworkError('An unexpected network failure occurred during analysis.');
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
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Demo Mode Banner (shown only when mode === 'demo') */}
      {isDemo && (
        <DemoModeBanner
          scenario={demoScenario}
          onScenarioChange={setDemoScenario}
        />
      )}

      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-black text-slate-900 tracking-tight">
          Educational Chest X-Ray Analysis
        </h1>
        <p className="text-sm text-slate-500 mt-1 max-w-2xl">
          Upload a de-identified educational chest radiograph for multi-stage triage inspection, quality validation, and explainability heatmaps.
        </p>
      </div>

      {/* Upload and Preview Section */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        <div className="lg:col-span-5 space-y-4">
          {!selectedFile ? (
            <ImageUploader onFileSelected={handleFileSelected} disabled={loading} />
          ) : (
            <div className="space-y-4">
              <ImagePreview
                file={selectedFile}
                onClear={handleClear}
                disabled={loading}
              />

              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={loading}
                  className="flex-1 inline-flex items-center justify-center gap-2 px-6 py-3 bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 text-white font-bold text-sm rounded-xl shadow-xs transition-colors focus:outline-none focus:ring-4 focus:ring-teal-500/20"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Analyzing Chest X-Ray...</span>
                    </>
                  ) : (
                    <>
                      <Search className="w-4 h-4" />
                      <span>Run Triage Analysis</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={handleClear}
                  disabled={loading}
                  className="px-4 py-3 bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium text-sm rounded-xl transition-colors focus:outline-none focus:ring-2 focus:ring-slate-300"
                >
                  Reset
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Results Region */}
        <div className="lg:col-span-7" aria-live="polite">
          {loading && (
            <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-xs">
              <div className="w-12 h-12 rounded-full border-4 border-teal-200 border-t-teal-600 animate-spin mx-auto mb-4"></div>
              <h3 className="text-base font-bold text-slate-800">
                Processing Radiograph
              </h3>
              <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
                Validating image dimensions, evaluating blur & contrast metrics, checking out-of-distribution distance, and computing Grad-CAM activations...
              </p>
            </div>
          )}

          {networkError && (
            <ErrorState
              message={networkError}
              onRetry={handleAnalyze}
            />
          )}

          {!loading && !analysisResult && !networkError && (
            <div className="bg-slate-100/60 rounded-2xl border-2 border-dashed border-slate-200 p-12 text-center text-slate-400">
              <Sliders className="w-8 h-8 mx-auto mb-2 opacity-60" />
              <p className="text-sm font-semibold text-slate-600">
                Awaiting Upload
              </p>
              <p className="text-xs text-slate-400 mt-1">
                Select or drop an educational chest radiograph to view triage support, confidence scores, and visual overlays.
              </p>
            </div>
          )}

          {!loading && analysisResult && (
            <div className="space-y-6">
              {/* Finding Card */}
              <FindingCard analysis={analysisResult} />

              {/* Confidence & Uncertainty */}
              <ConfidencePanel analysis={analysisResult} />

              {/* Quality & OOD Checks */}
              <QualityPanel
                quality={analysisResult.quality}
                ood={analysisResult.ood}
              />

              {/* Grad-CAM Heatmap Viewer */}
              <HeatmapViewer
                heatmap={analysisResult.heatmap}
                originalImageUrl={previewUrl}
                status={analysisResult.status}
              />

              {/* Evidence & Explanation */}
              <EvidencePanel
                explanation={analysisResult.explanation}
                evidence={analysisResult.evidence}
                limitations={analysisResult.limitations}
              />

              {/* Escalation & Triage Panel */}
              <EscalationPanel triage={analysisResult.triage} />

              {/* Model Metadata Footer */}
              <div className="p-4 bg-slate-100 rounded-xl border border-slate-200/80 text-[11px] text-slate-500 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-4 flex-wrap">
                  <span className="flex items-center gap-1">
                    <Database className="w-3.5 h-3.5 text-slate-400" />
                    <span>
                      Model: <strong>{analysisResult.model.model_name || 'N/A'}</strong> (
                      {analysisResult.model.model_version || 'v1'})
                    </span>
                  </span>
                  <span className="flex items-center gap-1">
                    <Tag className="w-3.5 h-3.5 text-slate-400" />
                    <span>
                      Dataset: <strong>{analysisResult.model.dataset || 'PneumoniaMNIST+'}</strong>
                    </span>
                  </span>
                </div>
                <div className="flex items-center gap-2 font-mono text-[10px] text-slate-400">
                  <Hash className="w-3 h-3" />
                  <span>Request: {analysisResult.request_id}</span>
                </div>
              </div>

              {/* Reset to analyze another image */}
              <div className="pt-2 text-center">
                <button
                  type="button"
                  onClick={handleClear}
                  className="inline-flex items-center gap-1.5 px-4 py-2 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold rounded-lg border border-slate-200 shadow-2xs transition-colors focus:outline-none focus:ring-2 focus:ring-teal-500"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Analyze Another Radiograph</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
