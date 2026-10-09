import React, { useState, useEffect, useRef } from 'react';
import {
  Bone,
  Upload,
  AlertTriangle,
  CheckCircle2,
  Sliders,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Info,
  ShieldAlert,
  Activity,
  Layers,
  Sparkles,
  ShieldCheck,
  RefreshCw,
  FileText
} from 'lucide-react';
import {
  getFractureStatus,
  predictFracture,
} from '../services/api';
import {
  FractureAnalysisResponse,
  FractureStatusResponse
} from '../types/analysis';
import { useToast } from './Toast';

export const BoneFractureWorkspace: React.FC = () => {
  const { showToast } = useToast();
  const [status, setStatus] = useState<FractureStatusResponse | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<FractureAnalysisResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Viewer controls
  const [activeTab, setActiveTab] = useState<'finding' | 'evidence' | 'quality' | 'tech'>('finding');
  const [showGradCam, setShowGradCam] = useState<boolean>(true);
  const [blendAlpha, setBlendAlpha] = useState<number>(0.5);
  const [zoom, setZoom] = useState<number>(1);
  const [invert, setInvert] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const fetchStatus = () => {
    getFractureStatus()
      .then((data) => setStatus(data))
      .catch(() => {
        setStatus({
          available: false,
          status: 'offline',
          message: 'Backend server offline on port 8000.'
        });
      });
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      loadFile(file);
    }
  };

  const loadFile = (file: File) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setErrorMsg(null);
    setZoom(1);
    setInvert(false);
    showToast(`Loaded ${file.name}`, 'info');
  };

  const loadPreset = async (url: string, name: string) => {
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], name, { type: blob.type || 'image/jpeg' });
      loadFile(file);
    } catch {
      showToast('Could not load preset radiograph asset', 'error');
    }
  };

  const handleClear = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setErrorMsg(null);
    setLoading(false);
  };

  const handleAnalyze = async () => {
    if (!selectedFile || loading) return;

    setLoading(true);
    setErrorMsg(null);
    setResult(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const resp = await predictFracture(selectedFile, controller.signal);
      setResult(resp);
      if (resp.status === 'model_unavailable' || resp.available === false) {
        setErrorMsg(resp.message || 'Model checkpoint is currently unavailable or training.');
        showToast('Model checkpoint unavailable', 'error');
      } else if (resp.fracture_detected) {
        showToast('Fracture-related finding identified', 'info');
      } else {
        showToast('No fracture-related findings identified', 'success');
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        const msg = err.message || 'Analysis failed. Connect to backend on port 8000.';
        setErrorMsg(msg);
        showToast(msg, 'error');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner: Model Information & Supported Regions */}
      <div className="bg-gradient-to-r from-slate-900 to-indigo-950 border border-indigo-900/60 rounded-xl p-5 shadow-lg">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
              <Bone className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-white tracking-tight">
                  Bone Fracture Analysis
                </h2>
                <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-400/30">
                  ConvNeXt-Base
                </span>
                <span className={`px-2 py-0.5 text-xs font-semibold rounded-full border ${
                  status?.available
                    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                    : 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                }`}>
                  {status?.available ? 'Model Online' : 'Check Status'}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Multi-region radiograph decision support powered by torchvision ConvNeXt-Base (88.6M parameters).
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="text-slate-400 mr-1 text-[11px] font-medium uppercase tracking-wider">Regions:</span>
            {['Wrist (Graz)', 'Hand', 'Leg', 'Hip', 'Shoulder', 'Mixed'].map((reg) => (
              <span
                key={reg}
                className="px-2 py-0.5 rounded bg-slate-800/80 text-slate-300 border border-slate-700/60 font-medium text-[11px]"
              >
                {reg}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Radiograph Viewer & Presets (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {/* Presets and Upload Controls */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                Diagnostic Presets
              </span>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/png,image/jpeg,image/jpg"
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="text-xs font-medium px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white flex items-center gap-1.5 transition-colors shadow-xs"
              >
                <Upload className="w-3.5 h-3.5" />
                Upload Radiograph
              </button>
            </div>

            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => loadPreset('/assets/sample-bone-wrist-fracture.png', 'wrist-fracture.png')}
                className="px-2.5 py-2 rounded-lg bg-slate-800/70 hover:bg-slate-700/80 border border-slate-700/50 text-left transition-colors group"
              >
                <div className="text-xs font-medium text-slate-200 group-hover:text-white truncate">Pediatric Wrist</div>
                <div className="text-[10px] text-amber-400 font-medium">Fracture Positive</div>
              </button>
              <button
                onClick={() => loadPreset('/assets/sample-bone-hand-fracture.jpg', 'hand-fracture.jpg')}
                className="px-2.5 py-2 rounded-lg bg-slate-800/70 hover:bg-slate-700/80 border border-slate-700/50 text-left transition-colors group"
              >
                <div className="text-xs font-medium text-slate-200 group-hover:text-white truncate">Hand Radiograph</div>
                <div className="text-[10px] text-amber-400 font-medium">Fracture Positive</div>
              </button>
              <button
                onClick={() => loadPreset('/assets/sample-bone-leg-normal.jpg', 'leg-normal.jpg')}
                className="px-2.5 py-2 rounded-lg bg-slate-800/70 hover:bg-slate-700/80 border border-slate-700/50 text-left transition-colors group"
              >
                <div className="text-xs font-medium text-slate-200 group-hover:text-white truncate">Lower Extremity</div>
                <div className="text-[10px] text-emerald-400 font-medium">No Findings</div>
              </button>
            </div>
          </div>

          {/* Interactive Viewer Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
            {/* Viewer Header / Toolbar */}
            <div className="px-4 py-2.5 bg-slate-950/60 border-b border-slate-800 flex items-center justify-between text-xs text-slate-400">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-slate-300">
                  {selectedFile ? selectedFile.name : 'No Radiograph Selected'}
                </span>
                {previewUrl && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                    Ready
                  </span>
                )}
              </div>

              {previewUrl && (
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setInvert(!invert)}
                    title="Invert Colors"
                    className={`p-1 rounded hover:bg-slate-800 ${invert ? 'text-indigo-400 bg-slate-800' : 'text-slate-400'}`}
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setZoom(Math.max(0.5, zoom - 0.25))}
                    title="Zoom Out"
                    className="p-1 rounded hover:bg-slate-800 text-slate-400"
                  >
                    <ZoomOut className="w-3.5 h-3.5" />
                  </button>
                  <span className="text-[10px] text-slate-400 font-mono">{Math.round(zoom * 100)}%</span>
                  <button
                    onClick={() => setZoom(Math.min(3, zoom + 0.25))}
                    title="Zoom In"
                    className="p-1 rounded hover:bg-slate-800 text-slate-400"
                  >
                    <ZoomIn className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={handleClear}
                    className="text-[11px] text-rose-400 hover:text-rose-300 font-medium ml-2"
                  >
                    Clear
                  </button>
                </div>
              )}
            </div>

            {/* Display Canvas Area */}
            <div className="relative h-[480px] bg-black/90 flex items-center justify-center overflow-hidden select-none">
              {previewUrl ? (
                <div
                  className="relative transition-transform duration-100 ease-out flex items-center justify-center max-h-full max-w-full"
                  style={{
                    transform: `scale(${zoom})`,
                    filter: invert ? 'invert(1)' : 'none'
                  }}
                >
                  {/* Original Radiograph */}
                  <img
                    src={previewUrl}
                    alt="Radiograph Preview"
                    className="max-h-[460px] w-auto object-contain rounded"
                  />

                  {/* Grad-CAM Heatmap Overlay */}
                  {result?.evidence?.gradcam_overlay_base64 && showGradCam && (
                    <img
                      src={result.evidence.gradcam_overlay_base64}
                      alt="Grad-CAM Overlay"
                      className="absolute inset-0 max-h-[460px] w-full h-full object-contain pointer-events-none transition-opacity duration-150"
                      style={{ opacity: blendAlpha }}
                    />
                  )}
                </div>
              ) : (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="flex flex-col items-center justify-center p-8 text-center cursor-pointer group"
                >
                  <div className="w-16 h-16 rounded-2xl bg-indigo-600/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 mb-3 group-hover:scale-105 transition-transform">
                    <Bone className="w-8 h-8" />
                  </div>
                  <h3 className="text-sm font-semibold text-slate-300 mb-1">
                    Upload or Select a Bone Radiograph
                  </h3>
                  <p className="text-xs text-slate-500 max-w-xs">
                    Supports high-resolution 16-bit and 8-bit PNG, JPEG formats across wrist, hand, leg, hip, and shoulder projections.
                  </p>
                </div>
              )}

              {/* In-Viewer Loading Overlay */}
              {loading && (
                <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-xs flex flex-col items-center justify-center text-white z-20">
                  <RefreshCw className="w-8 h-8 text-indigo-400 animate-spin mb-3" />
                  <span className="text-sm font-semibold text-slate-200">
                    Running ConvNeXt-Base Inference...
                  </span>
                  <span className="text-xs text-slate-400 mt-1">
                    Extracting 1024-d features & Grad-CAM visual heatmaps
                  </span>
                </div>
              )}
            </div>

            {/* Viewer Bottom Controls: Heatmap Slider & Analyze Button */}
            <div className="p-4 bg-slate-950/80 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-3">
              {result?.evidence?.gradcam_overlay_base64 ? (
                <div className="flex items-center gap-3 w-full sm:w-auto">
                  <label className="flex items-center gap-1.5 text-xs text-slate-300 font-medium cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showGradCam}
                      onChange={(e) => setShowGradCam(e.target.checked)}
                      className="rounded border-slate-700 text-indigo-600 focus:ring-indigo-500"
                    />
                    <Layers className="w-3.5 h-3.5 text-indigo-400" />
                    Grad-CAM Overlay
                  </label>
                  {showGradCam && (
                    <div className="flex items-center gap-2">
                      <Sliders className="w-3.5 h-3.5 text-slate-400" />
                      <input
                        type="range"
                        min="0.1"
                        max="1.0"
                        step="0.05"
                        value={blendAlpha}
                        onChange={(e) => setBlendAlpha(parseFloat(e.target.value))}
                        className="w-24 accent-indigo-500"
                      />
                      <span className="text-[11px] text-slate-400 font-mono">{Math.round(blendAlpha * 100)}%</span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-xs text-slate-500 flex items-center gap-1.5">
                  <Info className="w-3.5 h-3.5" />
                  Select or upload an image to begin automated triage
                </div>
              )}

              <button
                disabled={!selectedFile || loading}
                onClick={handleAnalyze}
                className="w-full sm:w-auto px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-md hover:shadow-indigo-500/20"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Activity className="w-3.5 h-3.5" />
                    Analyze Bone Radiograph
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Multi-Tab Inspection & Evidence Panel (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
            {/* Inspection Tabs Header */}
            <div className="flex border-b border-slate-800 bg-slate-950/60 p-1">
              {[
                { id: 'finding', label: 'Findings', icon: Activity },
                { id: 'evidence', label: 'Evidence', icon: Layers },
                { id: 'quality', label: 'Quality', icon: ShieldCheck },
                { id: 'tech', label: 'Validation', icon: FileText },
              ].map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`flex-1 py-2 text-xs font-semibold rounded-lg flex items-center justify-center gap-1.5 transition-colors ${
                      isActive
                        ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    {tab.label}
                  </button>
                );
              })}
            </div>

            {/* Inspection Tab Body */}
            <div className="p-5 min-h-[460px] flex flex-col justify-between space-y-4">
              {/* TAB 1: FINDINGS */}
              {activeTab === 'finding' && (
                <div className="space-y-4">
                  {result && result.available ? (
                    <>
                      {/* Diagnostic Result Banner */}
                      <div className={`p-4 rounded-xl border ${
                        result.fracture_detected
                          ? 'bg-rose-950/40 border-rose-500/40 text-rose-200'
                          : 'bg-emerald-950/40 border-emerald-500/40 text-emerald-200'
                      }`}>
                        <div className="flex items-start gap-3">
                          {result.fracture_detected ? (
                            <AlertTriangle className="w-6 h-6 text-rose-400 flex-shrink-0 mt-0.5" />
                          ) : (
                            <CheckCircle2 className="w-6 h-6 text-emerald-400 flex-shrink-0 mt-0.5" />
                          )}
                          <div className="space-y-1">
                            <h4 className="text-sm font-bold tracking-tight">
                              {result.finding}
                            </h4>
                            <p className="text-xs opacity-90 leading-relaxed">
                              {result.fracture_detected
                                ? 'Model identified high-activation textural/structural features consistent with fracture pathology.'
                                : 'No overt fracture indicators detected in this single radiographic projection.'}
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Probabilities and Calibrated Decision */}
                      <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-3">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-slate-400">Calibrated Probability</span>
                          <span className="font-mono font-bold text-white text-sm">
                            {(result.calibrated_probability! * 100).toFixed(1)}%
                          </span>
                        </div>
                        {/* Progress bar */}
                        <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              result.fracture_detected ? 'bg-rose-500' : 'bg-emerald-500'
                            }`}
                            style={{ width: `${result.calibrated_probability! * 100}%` }}
                          />
                        </div>

                        <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800/80 text-[11px]">
                          <div>
                            <span className="text-slate-500 block">Threshold:</span>
                            <span className="font-mono text-slate-300">
                              {(result.decision_threshold! * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div>
                            <span className="text-slate-500 block">Uncertainty:</span>
                            <span className={`font-semibold capitalize ${
                              result.uncertainty === 'high'
                                ? 'text-rose-400'
                                : result.uncertainty === 'medium'
                                ? 'text-amber-400'
                                : 'text-emerald-400'
                            }`}>
                              {result.uncertainty}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Clinical Review Guidance Banner */}
                      <div className="bg-indigo-950/30 border border-indigo-900/50 rounded-xl p-3.5 space-y-1.5 text-xs text-indigo-300">
                        <div className="flex items-center gap-2 font-semibold">
                          <ShieldAlert className="w-4 h-4 text-indigo-400" />
                          Clinical Decision Support Guidance
                        </div>
                        <p className="text-[11px] text-slate-300 leading-relaxed">
                          {result.review_required
                            ? 'Mandatory secondary review recommended. Correlate with clinical tenderness, swelling, and multi-angle views.'
                            : 'AI findings are negative; however, occult fractures (e.g. scaphoid, hair-line pediatric fissures) can appear radiographically silent.'}
                        </p>
                      </div>
                    </>
                  ) : errorMsg ? (
                    <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-600/40 text-amber-200 space-y-2">
                      <div className="flex items-center gap-2 font-bold text-sm">
                        <AlertTriangle className="w-4 h-4 text-amber-400" />
                        Model Unavailable
                      </div>
                      <p className="text-xs text-slate-300">{errorMsg}</p>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-slate-500 space-y-3">
                      <Bone className="w-10 h-10 mx-auto text-slate-600" />
                      <div className="text-xs font-medium">Awaiting Radiograph Analysis</div>
                      <p className="text-[11px] text-slate-500 max-w-xs mx-auto">
                        Upload or select a preset radiograph to trigger real-time ConvNeXt-Base fracture detection.
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* TAB 2: VISUAL EVIDENCE */}
              {activeTab === 'evidence' && (
                <div className="space-y-4">
                  {result?.evidence?.gradcam_overlay_base64 ? (
                    <div className="space-y-3">
                      <div className="rounded-xl overflow-hidden border border-slate-800 bg-black">
                        <img
                          src={result.evidence.gradcam_overlay_base64}
                          alt="Grad-CAM Focus"
                          className="w-full h-48 object-contain"
                        />
                      </div>
                      <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3.5 text-xs text-slate-300 space-y-2">
                        <div className="font-semibold text-slate-200 flex items-center gap-1.5">
                          <Layers className="w-4 h-4 text-indigo-400" />
                          Visual Attribution (Grad-CAM)
                        </div>
                        <p className="text-[11px] text-slate-400 leading-relaxed">
                          {result.evidence.disclaimer}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-slate-500 space-y-2">
                      <Layers className="w-8 h-8 mx-auto text-slate-600" />
                      <div className="text-xs">No Visual Evidence Generated Yet</div>
                    </div>
                  )}
                </div>
              )}

              {/* TAB 3: QUALITY & ANATOMY */}
              {activeTab === 'quality' && (
                <div className="space-y-4 text-xs">
                  {result?.image_quality ? (
                    <div className="space-y-3">
                      <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3.5 space-y-2">
                        <span className="font-semibold text-slate-200 block">Image Quality Metrics</span>
                        <div className="grid grid-cols-2 gap-2 text-[11px]">
                          <div>
                            <span className="text-slate-500 block">Resolution:</span>
                            <span className="font-mono text-slate-300">
                              {result.image_quality.width} × {result.image_quality.height}
                            </span>
                          </div>
                          <div>
                            <span className="text-slate-500 block">Contrast (StdDev):</span>
                            <span className="font-mono text-slate-300">
                              {result.image_quality.std_intensity}
                            </span>
                          </div>
                          <div>
                            <span className="text-slate-500 block">Mean Intensity:</span>
                            <span className="font-mono text-slate-300">
                              {result.image_quality.mean_intensity}
                            </span>
                          </div>
                          <div>
                            <span className="text-slate-500 block">Quality Status:</span>
                            <span className="font-semibold text-emerald-400 capitalize">
                              {result.image_quality.status}
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3.5 space-y-2">
                        <span className="font-semibold text-slate-200 block">Supported Body Regions</span>
                        <div className="flex flex-wrap gap-1">
                          {result.supported_anatomies?.map((a) => (
                            <span key={a} className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-[10px]">
                              {a}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-slate-500">
                      <ShieldCheck className="w-8 h-8 mx-auto text-slate-600 mb-2" />
                      <div className="text-xs">Quality Assessment Available Upon Analysis</div>
                    </div>
                  )}
                </div>
              )}

              {/* TAB 4: TECHNICAL SPECIFICATION */}
              {activeTab === 'tech' && (
                <div className="space-y-3 text-xs">
                  <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3.5 space-y-2 text-[11px]">
                    <span className="font-semibold text-slate-200 block text-xs">Model Architecture</span>
                    <div className="space-y-1.5 text-slate-400">
                      <div className="flex justify-between">
                        <span>Backbone:</span>
                        <span className="text-slate-200 font-mono">ConvNeXt-Base</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Parameters:</span>
                        <span className="text-slate-200 font-mono">88.6 Million</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Feature Dimension:</span>
                        <span className="text-slate-200 font-mono">1024 dims</span>
                      </div>
                      <div className="flex justify-between">
                        <span>MLP Head:</span>
                        <span className="text-slate-200 font-mono">1024 → 512 → 256 → 1</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Loss Function:</span>
                        <span className="text-slate-200 font-mono">Weighted BCEWithLogits</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3.5 space-y-1 text-[11px] text-slate-400">
                    <span className="font-semibold text-slate-200 block text-xs">Evaluation Benchmarks</span>
                    <p className="leading-relaxed">
                      Evaluated against Graz Pediatric Wrist Radiographs (20,327 images) and FracAtlas Multi-Region benchmark (4,083 images) with zero patient leakage.
                    </p>
                  </div>
                </div>
              )}

              {/* Research Protocol Disclaimer Footer */}
              <div className="pt-3 border-t border-slate-800/80 text-[10px] text-slate-500 text-center leading-normal">
                MedGuard AI is an educational clinical triage & research tool. Not certified for definitive diagnosis or treatment without physician confirmation.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
