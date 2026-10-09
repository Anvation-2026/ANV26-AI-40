import React, { useEffect, useRef, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Search,
  Loader2,
  Bone,
  Activity,
} from 'lucide-react';
import { XrayViewer } from '../components/XrayViewer';
import { InspectorPanel, InspectorTab } from '../components/InspectorPanel';
import { DemoModeBanner } from '../components/DemoModeBanner';
import { BoneFractureWorkspace } from '../components/BoneFractureWorkspace';
import { useToast } from '../components/Toast';
import { getModelStatus, predictImage, NetworkError } from '../services/api';
import { saveRecentAnalysis } from '../services/history';
import { loadSampleAsset, SAMPLE_PRESETS } from '../services/sampleAssets';
import { AnalysisResponse, ModelStatusResponse } from '../types/analysis';

export const Workspace: React.FC = () => {
  const { showToast } = useToast();
  const [analysisModality, setAnalysisModality] = useState<'chest' | 'fracture'>('chest');
  const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(null);
  const [demoScenario, setDemoScenario] = useState<string>('uncertain');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [networkError, setNetworkError] = useState<string | null>(null);
  const [activeInspectorTab, setActiveInspectorTab] = useState<InspectorTab>('finding');
  const [viewMode, setViewMode] = useState<'overlay' | 'original' | 'split' | 'side_by_side'>('overlay');

  const abortControllerRef = useRef<AbortController | null>(null);
  const timeoutIdRef = useRef<number | null>(null);

  const fetchStatus = useCallback(() => {
    getModelStatus()
      .then((data) => setModelStatus(data))
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchStatus();
    return () => {
      if (abortControllerRef.current) abortControllerRef.current.abort();
      if (timeoutIdRef.current) clearTimeout(timeoutIdRef.current);
    };
  }, [fetchStatus]);

  const handleFileSelected = (file: File) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setAnalysisResult(null);
    setNetworkError(null);
    setActiveInspectorTab('finding');
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
    setActiveInspectorTab('finding');
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
      setActiveInspectorTab('finding');

      // Save to recent session history
      saveRecentAnalysis(selectedFile.name, res, previewUrl || undefined);

      if (res.status === 'success') {
        showToast('Decision support evaluation complete', 'success');
      } else if (res.status === 'uncertain') {
        showToast('Decision abstained due to uncertainty', 'info');
      } else if (res.status === 'poor_quality') {
        showToast('Image rejected: quality check failed', 'info');
      } else if (res.status === 'ood') {
        showToast('Image rejected: out-of-distribution input', 'info');
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
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="space-y-3.5 h-full flex flex-col"
    >
      {/* Modality Selector Navigation */}
      <div className="flex items-center justify-between border-b border-border pb-2 flex-shrink-0">
        <div className="flex items-center gap-1.5 p-1 bg-surface rounded-xl border border-border shadow-xs">
          <button
            type="button"
            onClick={() => setAnalysisModality('chest')}
            className={`px-3.5 py-1.5 text-xs font-semibold rounded-lg flex items-center gap-2 transition-all ${
              analysisModality === 'chest'
                ? 'bg-teal-600 text-white shadow-xs'
                : 'text-navy-muted hover:text-navy-foreground'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            Chest Radiographs (Pneumonia, TB &amp; Chest-14)
          </button>
          <button
            type="button"
            onClick={() => setAnalysisModality('fracture')}
            className={`px-3.5 py-1.5 text-xs font-semibold rounded-lg flex items-center gap-2 transition-all ${
              analysisModality === 'fracture'
                ? 'bg-indigo-600 text-white shadow-xs'
                : 'text-navy-muted hover:text-navy-foreground'
            }`}
          >
            <Bone className="w-3.5 h-3.5" />
            Bone Fracture Analysis (ConvNeXt-Base)
          </button>
        </div>
      </div>

      {analysisModality === 'fracture' ? (
        <div className="flex-1 overflow-y-auto">
          <BoneFractureWorkspace />
        </div>
      ) : (
        <>
          {/* Top Clinical Operational Toolbar */}
      <div className="bg-surface rounded-[12px] border border-border px-4 py-2.5 shadow-xs flex flex-wrap items-center justify-between gap-3 text-xs flex-shrink-0">
        {/* Left: Branding & Status */}
        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-teal-50 border border-teal-200/80 text-teal-800 text-[11px] font-semibold">
            <span className="w-1.5 h-1.5 rounded-full bg-teal-600 animate-pulse" />
            <span>Workstation Active</span>
          </div>

          <span className="text-slate-300 hidden sm:inline">&middot;</span>

          <span className="text-xs text-navy-muted hidden sm:inline">
            PneumoniaMNIST+ (224&times;224) &middot; Independent Quality, Uncertainty &amp; Grad-CAM
          </span>
        </div>

        {/* Right: Quick Samples & Primary Actions */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-1 flex-wrap">
            <span className="text-[11px] font-medium text-navy-muted mr-1 hidden lg:inline">
              Samples:
            </span>
            {SAMPLE_PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => loadSamplePreset(preset.assetUrl, preset.filename)}
                disabled={loading}
                className={`px-2.5 py-1 text-[11px] font-medium active:scale-95 border rounded-[6px] transition-all ${
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

          <div className="h-4 w-px bg-border hidden sm:block mx-0.5" />

          {/* Action Trigger Buttons */}
          {selectedFile && (
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={loading}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 bg-teal-700 hover:bg-teal-800 active:scale-95 disabled:bg-slate-300 text-white font-semibold text-xs rounded-[7px] shadow-xs transition-all"
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
                className="px-2.5 py-1.5 bg-canvas hover:bg-slate-200/80 text-navy-muted hover:text-navy-foreground text-xs font-medium rounded-[7px] border border-border transition-colors"
                title="Clear current radiograph and results"
              >
                Clear
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Demo Mode Notice (only rendered if backend is in demo mode) */}
      {isDemo && (
        <DemoModeBanner
          scenario={demoScenario}
          onScenarioChange={setDemoScenario}
        />
      )}

      {/* Main Radiology Workstation Grid: 62% Left Canvas / 38% Right Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1 min-h-[580px] items-stretch">
        {/* Left Column: 62% Radiology Canvas (always visible) */}
        <div className="lg:col-span-7 xl:col-span-8 flex flex-col h-full min-h-[500px]">
          <XrayViewer
            originalUrl={previewUrl}
            heatmap={analysisResult?.heatmap || null}
            status={analysisResult?.status}
            onFileSelected={handleFileSelected}
            onLoadSample={loadSamplePreset}
            loading={loading}
            onClear={handleClear}
            fileName={selectedFile?.name}
            fileSize={selectedFile?.size}
            viewMode={viewMode}
            onViewModeChange={setViewMode}
            className="flex-1"
          />
        </div>

        {/* Right Column: 38% Tabbed Clinical Inspector (Independently Scrollable) */}
        <div className="lg:col-span-5 xl:col-span-4 flex flex-col h-full min-h-[500px] overflow-hidden">
          <InspectorPanel
            analysis={analysisResult}
            loading={loading}
            networkError={networkError}
            selectedFile={selectedFile}
            onAnalyze={handleAnalyze}
            onClear={handleClear}
            activeTab={activeInspectorTab}
            onTabChange={setActiveInspectorTab}
            onFocusHeatmap={() => {
              setViewMode('overlay');
              showToast('Overlaying ambiguity attention highlight', 'info');
            }}
          />
        </div>
      </div>
        </>
      )}

      {/* Minimal System Disclaimer Footer */}
      <div className="text-center py-1 text-[11px] text-navy-muted/80 flex-shrink-0">
        MedGuard AI is an educational research prototype &middot; Non-diagnostic &middot; Not cleared for clinical management
      </div>
    </motion.div>
  );
};
