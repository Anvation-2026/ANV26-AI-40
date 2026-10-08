import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Layers,
  Columns,
  Eye,
  Info,
  Maximize2,
  Minimize2,
  Move,
  SlidersHorizontal,
  SplitSquareVertical,
  UploadCloud,
  FileUp,
} from 'lucide-react';
import { HeatmapInfo } from '../types/analysis';

export interface XrayViewerProps {
  originalUrl: string | null;
  heatmap: HeatmapInfo | null;
  status?: string;
  className?: string;
  onFileSelected?: (file: File) => void;
  onLoadSample?: (url: string, name: string) => void;
  loading?: boolean;
  onClear?: () => void;
  fileName?: string;
  fileSize?: number;
  viewMode?: 'overlay' | 'original' | 'split' | 'side_by_side';
  onViewModeChange?: (mode: 'overlay' | 'original' | 'split' | 'side_by_side') => void;
}

const MAX_BYTES = 10 * 1024 * 1024;
const ALLOWED_MIME = ['image/png', 'image/jpeg', 'image/webp'];

export const XrayViewer: React.FC<XrayViewerProps> = ({
  originalUrl,
  heatmap,
  status,
  className = '',
  onFileSelected,
  onLoadSample,
  loading = false,
  fileName,
  fileSize,
  viewMode: controlledViewMode,
  onViewModeChange,
}) => {
  const [scale, setScale] = useState<number>(1);
  const [position, setPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [internalViewMode, setInternalViewMode] = useState<'overlay' | 'original' | 'split' | 'side_by_side'>('overlay');
  
  const viewMode = controlledViewMode ?? internalViewMode;
  const setViewMode = useCallback((mode: 'overlay' | 'original' | 'split' | 'side_by_side') => {
    setInternalViewMode(mode);
    onViewModeChange?.(mode);
  }, [onViewModeChange]);

  const [opacity, setOpacity] = useState<number>(0.65);
  const [splitPosition, setSplitPosition] = useState<number>(50); // percentage 0-100
  const [isDraggingSplit, setIsDraggingSplit] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [isDragOverDropzone, setIsDragOverDropzone] = useState<boolean>(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const hasHeatmap = Boolean(
    (heatmap?.available || Boolean(heatmap?.data_url)) &&
    heatmap?.data_url?.startsWith('data:image/png;base64,')
  );
  const isRejectedOrAbstained = (status === 'poor_quality' || status === 'ood') || (status === 'uncertain' && !hasHeatmap);

  const resetView = useCallback(() => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  }, []);

  const handleZoom = (delta: number) => {
    setScale((prev) => {
      const next = Math.min(Math.max(Number((prev + delta).toFixed(2)), 0.75), 4.0);
      if (next === 1) setPosition({ x: 0, y: 0 });
      return next;
    });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (isDraggingSplit) return;
    if (scale <= 1) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDraggingSplit && viewportRef.current) {
      const rect = viewportRef.current.getBoundingClientRect();
      const relativeX = e.clientX - rect.left;
      const percentage = Math.min(Math.max((relativeX / rect.width) * 100, 5), 95);
      setSplitPosition(percentage);
      return;
    }

    if (!isDragging || scale <= 1) return;
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setIsDraggingSplit(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      handleZoom(e.deltaY < 0 ? 0.2 : -0.2);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0] && onFileSelected) {
      onFileSelected(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOverDropzone(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0] && onFileSelected) {
      const file = e.dataTransfer.files[0];
      const isExtensionMatch = /\.(png|jpe?g|webp)$/i.test(file.name);
      const isMimeMatch = ALLOWED_MIME.includes(file.type);
      if ((isExtensionMatch || isMimeMatch) && file.size <= MAX_BYTES) {
        onFileSelected(file);
      }
    }
  };

  useEffect(() => {
    resetView();
    if (hasHeatmap) {
      setViewMode('overlay');
    } else {
      setViewMode('original');
    }
  }, [originalUrl, heatmap?.data_url, hasHeatmap, resetView]);

  // Handle Esc key in fullscreen
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isFullscreen]);

  return (
    <div
      ref={containerRef}
      className={`relative flex flex-col bg-[#07111b] border border-[#1b2b3a] rounded-[12px] overflow-hidden select-none transition-all shadow-md ${
        isFullscreen ? 'fixed inset-3 z-50 rounded-xl' : 'w-full h-full min-h-[500px]'
      } ${className}`}
      onWheel={handleWheel}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileInputChange}
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
      />

      {/* Top Clinical Toolbar (Linear / OHIF style) */}
      <div className="flex items-center justify-between px-3 py-2 bg-[#0d1d2b]/95 border-b border-white/10 backdrop-blur z-20 text-white text-xs gap-2 flex-wrap">
        {/* Left: View Mode Toggles */}
        <div className="flex items-center gap-1 bg-[#07131e] p-1 rounded-[8px] border border-white/5">
          <button
            type="button"
            onClick={() => setViewMode('overlay')}
            className={`px-2.5 py-1 rounded-[6px] font-medium transition-colors flex items-center gap-1.5 ${
              viewMode === 'overlay' ? 'bg-teal-600 text-white shadow-xs' : 'text-slate-300 hover:text-white'
            }`}
            title="Heatmap overlay blend"
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Overlay</span>
          </button>

          {hasHeatmap && originalUrl && (
            <button
              type="button"
              onClick={() => setViewMode('split')}
              className={`px-2.5 py-1 rounded-[6px] font-medium transition-colors flex items-center gap-1.5 ${
                viewMode === 'split' ? 'bg-teal-600 text-white shadow-xs' : 'text-slate-300 hover:text-white'
              }`}
              title="Split comparison slider"
            >
              <SplitSquareVertical className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Split Slider</span>
            </button>
          )}

          {hasHeatmap && originalUrl && (
            <button
              type="button"
              onClick={() => setViewMode('side_by_side')}
              className={`px-2.5 py-1 rounded-[6px] font-medium transition-colors flex items-center gap-1.5 ${
                viewMode === 'side_by_side' ? 'bg-teal-600 text-white shadow-xs' : 'text-slate-300 hover:text-white'
              }`}
              title="Side-by-side original and Grad-CAM"
            >
              <Columns className="w-3.5 h-3.5" />
              <span className="hidden md:inline">Dual</span>
            </button>
          )}

          {originalUrl && (
            <button
              type="button"
              onClick={() => setViewMode('original')}
              className={`px-2.5 py-1 rounded-[6px] font-medium transition-colors flex items-center gap-1.5 ${
                viewMode === 'original' ? 'bg-teal-600 text-white shadow-xs' : 'text-slate-300 hover:text-white'
              }`}
              title="Show raw original radiograph"
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Original</span>
            </button>
          )}
        </div>

        {/* Middle: Opacity Slider (when Overlay active and heatmap exists) */}
        {hasHeatmap && viewMode === 'overlay' && (
          <div className="hidden sm:flex items-center gap-2 bg-[#07131e] px-2.5 py-1 rounded-[8px] border border-white/5">
            <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-[11px] text-slate-400">Heatmap:</span>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-18 sm:w-24 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-400"
              title="Adjust heatmap transparency"
            />
            <span className="font-mono text-[10px] text-teal-400 tabular-nums">
              {Math.round(opacity * 100)}%
            </span>
          </div>
        )}

        {/* Right: Zoom & Canvas Actions */}
        <div className="flex items-center gap-1.5">
          {originalUrl && onFileSelected && (
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="px-2 py-1 rounded-[6px] bg-[#07131e] hover:bg-[#122434] text-slate-300 hover:text-white border border-white/10 text-[11px] font-medium flex items-center gap-1.5 transition-colors"
              title="Replace current radiograph"
            >
              <FileUp className="w-3 h-3 text-teal-400" />
              <span>Replace</span>
            </button>
          )}

          <div className="flex items-center bg-[#07131e] rounded-[8px] p-0.5 border border-white/5">
            <button
              type="button"
              onClick={() => handleZoom(-0.25)}
              disabled={!originalUrl || scale <= 0.75}
              className="p-1.5 rounded-[6px] hover:bg-white/10 disabled:opacity-30 disabled:hover:bg-transparent text-slate-300 hover:text-white transition-colors"
              title="Zoom Out (Ctrl -)"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>

            <span className="px-1.5 font-mono text-[11px] text-slate-400 tabular-nums min-w-[42px] text-center">
              {Math.round(scale * 100)}%
            </span>

            <button
              type="button"
              onClick={() => handleZoom(0.25)}
              disabled={!originalUrl || scale >= 4.0}
              className="p-1.5 rounded-[6px] hover:bg-white/10 disabled:opacity-30 disabled:hover:bg-transparent text-slate-300 hover:text-white transition-colors"
              title="Zoom In (Ctrl +)"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
          </div>

          <button
            type="button"
            onClick={resetView}
            disabled={!originalUrl || (scale === 1 && position.x === 0 && position.y === 0)}
            className="p-1.5 rounded-[8px] bg-[#07131e] hover:bg-white/10 disabled:opacity-30 text-slate-300 hover:text-white border border-white/5 transition-colors"
            title="Reset Pan & Zoom"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1.5 rounded-[8px] bg-[#07131e] hover:bg-white/10 text-slate-300 hover:text-white border border-white/5 transition-colors"
            title={isFullscreen ? 'Exit Fullscreen (Esc)' : 'Expand Fullscreen'}
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Main Radiology Canvas Area */}
      <div
        ref={viewportRef}
        className={`relative flex-1 flex items-center justify-center overflow-hidden bg-[#04090f] ${
          scale > 1 ? 'cursor-grab active:cursor-grabbing' : 'cursor-default'
        }`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onDragOver={(e) => {
          e.preventDefault();
          if (!originalUrl) setIsDragOverDropzone(true);
        }}
        onDragLeave={() => setIsDragOverDropzone(false)}
        onDrop={handleDrop}
      >
        <div
          className="absolute inset-0 pointer-events-none opacity-10"
          style={{
            backgroundImage:
              'linear-gradient(to right, #1a2b3c 1px, transparent 1px), linear-gradient(to bottom, #1a2b3c 1px, transparent 1px)',
            backgroundSize: '32px 32px',
          }}
        />

        {/* Honest Loading Overlay during inference */}
        {loading && (
          <div className="absolute inset-0 z-30 bg-[#07111b]/80 backdrop-blur-xs flex flex-col items-center justify-center text-white gap-3 animate-fadeIn">
            <div className="w-10 h-10 rounded-full border-3 border-teal-500/20 border-t-teal-400 animate-spin" />
            <div className="text-center">
              <span className="text-xs font-bold tracking-tight text-white block">
                Executing Inference Pipeline
              </span>
              <span className="text-[11px] text-slate-400 mt-0.5 block">
                Evaluating ResNet-18 forward pass &amp; Grad-CAM saliency...
              </span>
            </div>
          </div>
        )}

        {/* Ambiguity Attention Map floating badge when uncertain and heatmap is active */}
        {hasHeatmap && status === 'uncertain' && viewMode !== 'original' && originalUrl && (
          <div className="absolute top-3 left-3 z-20 pointer-events-none animate-fadeIn">
            <div className="bg-[#121c27]/90 border border-amber-500/40 text-amber-200 text-[11px] px-3 py-1.5 rounded-[8px] backdrop-blur flex items-center gap-2 shadow-lg">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              <span>
                <strong>Ambiguity Attention Map:</strong> Highlighting subtle lung regions influencing borderline score
              </span>
            </div>
          </div>
        )}

        {/* View Mode: Split Comparison Slider */}
        {hasHeatmap && viewMode === 'split' && originalUrl ? (
          <div className="relative w-full h-full flex items-center justify-center p-4">
            <div
              className="relative max-h-[500px] w-auto max-w-full rounded-[8px] overflow-hidden shadow-2xl border border-white/10 select-none"
              style={{
                transform: `scale(${scale}) translate(${position.x / scale}px, ${position.y / scale}px)`,
                transformOrigin: 'center center',
              }}
            >
              <img
                src={originalUrl}
                alt="Original radiograph"
                className="max-h-[500px] w-auto max-w-full object-contain pointer-events-none block"
              />

              <div
                className="absolute inset-0 overflow-hidden pointer-events-none"
                style={{ width: `${splitPosition}%` }}
              >
                <img
                  src={originalUrl}
                  alt="Original underlay"
                  className="max-h-[500px] w-auto max-w-full object-contain"
                />
                <img
                  src={heatmap?.data_url || ''}
                  alt="Grad-CAM overlay clip"
                  style={{ opacity: 0.85 }}
                  className="absolute inset-0 w-full h-full object-contain mix-blend-screen"
                />
              </div>

              <div
                className="absolute top-0 bottom-0 z-20 cursor-ew-resize flex items-center justify-center"
                style={{ left: `${splitPosition}%`, transform: 'translateX(-50%)' }}
                onMouseDown={(e) => {
                  e.stopPropagation();
                  setIsDraggingSplit(true);
                }}
              >
                <div className="w-0.5 h-full bg-teal-400 shadow-[0_0_8px_rgba(45,212,191,0.8)]" />
                <div className="absolute w-6 h-6 rounded-full bg-teal-500 border-2 border-white shadow-lg flex items-center justify-center text-[10px] text-white">
                  &harr;
                </div>
              </div>

              <div className="absolute top-2 left-2 z-10 bg-black/70 px-2 py-0.5 rounded text-[10px] font-mono text-teal-300">
                Grad-CAM Activation
              </div>
              <div className="absolute top-2 right-2 z-10 bg-black/70 px-2 py-0.5 rounded text-[10px] font-mono text-slate-300">
                Original Radiograph
              </div>
            </div>
          </div>
        ) : hasHeatmap && viewMode === 'side_by_side' && originalUrl ? (
          /* Dual Side-by-Side Mode */
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 w-full h-full items-center justify-items-center overflow-auto">
            <div className="relative flex flex-col items-center">
              <span className="text-[10px] font-mono text-slate-400 mb-1.5 uppercase tracking-wider">
                Original AP Radiograph
              </span>
              <img
                src={originalUrl}
                alt="Original radiograph"
                className="max-h-[400px] w-auto max-w-full rounded-[8px] shadow-lg border border-white/10 object-contain pointer-events-none"
              />
            </div>
            <div className="relative flex flex-col items-center">
              <span className="text-[10px] font-mono text-teal-400 mb-1.5 uppercase tracking-wider">
                Grad-CAM Activation Saliency
              </span>
              <img
                src={heatmap?.data_url || ''}
                alt="Grad-CAM activation heatmap"
                className="max-h-[400px] w-auto max-w-full rounded-[8px] shadow-lg border border-teal-500/20 object-contain pointer-events-none"
              />
            </div>
          </div>
        ) : (
          /* Single Image / Overlay Mode */
          <div
            className="relative flex items-center justify-center transition-transform"
            style={{
              transform: `scale(${scale}) translate(${position.x / scale}px, ${position.y / scale}px)`,
              transformOrigin: 'center center',
            }}
          >
            {originalUrl ? (
              <div className="relative">
                <img
                  src={originalUrl}
                  alt="Chest radiograph preview"
                  className="max-h-[460px] w-auto max-w-full rounded-[8px] shadow-2xl object-contain border border-white/5 pointer-events-none"
                />

                {hasHeatmap && viewMode === 'overlay' && (
                  <img
                    src={heatmap?.data_url || ''}
                    alt="Grad-CAM activation overlay"
                    style={{ opacity }}
                    className="absolute inset-0 w-full h-full object-contain rounded-[8px] pointer-events-none mix-blend-screen"
                  />
                )}
              </div>
            ) : (
              /* Compact Dropzone Inside Viewer when empty */
              <div
                className={`flex flex-col items-center justify-center p-8 text-center max-w-md rounded-[12px] border-2 border-dashed transition-all ${
                  isDragOverDropzone
                    ? 'border-teal-400 bg-teal-950/30 scale-[1.02]'
                    : 'border-white/15 bg-[#0a1624]/60 hover:border-teal-500/50'
                }`}
              >
                <div className="w-12 h-12 rounded-xl bg-teal-950/80 border border-teal-500/30 flex items-center justify-center text-teal-400 mb-3.5 shadow-inner">
                  <UploadCloud className="w-6 h-6" />
                </div>
                <h3 className="text-sm font-bold text-white tracking-tight">
                  Upload Chest Radiograph
                </h3>
                <p className="text-xs text-slate-400 mt-1 max-w-xs leading-relaxed">
                  Drag and drop a chest X-ray image here, or browse from your computer.
                </p>

                <div className="flex items-center gap-2 mt-4">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="px-4 py-2 bg-teal-600 hover:bg-teal-500 active:scale-95 text-white text-xs font-semibold rounded-[8px] shadow-sm transition-all"
                  >
                    Browse Files
                  </button>
                </div>

                <span className="text-[10px] text-slate-500 mt-2.5 block">
                  PNG, JPEG, WEBP &le; 10 MB &middot; De-identified for educational use
                </span>

                {/* Quick Educational Sample Chips */}
                {onLoadSample && (
                  <div className="mt-5 pt-4 border-t border-white/10 w-full">
                    <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-2">
                      Or Load Educational Sample
                    </span>
                    <div className="flex items-center justify-center gap-1.5 flex-wrap">
                      <button
                        type="button"
                        onClick={() => onLoadSample('/assets/sample-normal.png', 'sample-normal.png')}
                        className="px-2.5 py-1 text-[11px] font-medium text-slate-200 bg-[#122434] hover:bg-teal-900/60 hover:text-teal-300 border border-white/10 rounded-[6px] transition-all"
                      >
                        Normal
                      </button>
                      <button
                        type="button"
                        onClick={() => onLoadSample('/assets/sample-pneumonia.png', 'sample-pneumonia.png')}
                        className="px-2.5 py-1 text-[11px] font-medium text-slate-200 bg-[#122434] hover:bg-teal-900/60 hover:text-teal-300 border border-white/10 rounded-[6px] transition-all"
                      >
                        Pneumonia
                      </button>
                      <button
                        type="button"
                        onClick={() => onLoadSample('/assets/demo-uncertain.png', 'sample-uncertain.png')}
                        className="px-2.5 py-1 text-[11px] font-medium text-amber-300 bg-[#122434] hover:bg-amber-950/60 hover:text-amber-200 border border-amber-500/30 rounded-[6px] transition-all"
                      >
                        Uncertain
                      </button>
                      <button
                        type="button"
                        onClick={() => onLoadSample('/assets/demo-blurred.png', 'sample-blurred.png')}
                        className="px-2.5 py-1 text-[11px] font-medium text-slate-400 bg-[#122434] hover:bg-slate-700/60 hover:text-slate-200 border border-white/10 rounded-[6px] transition-all"
                      >
                        Blur Test
                      </button>
                      <button
                        type="button"
                        onClick={() => onLoadSample('/assets/demo-ood.png', 'sample-ood.png')}
                        className="px-2.5 py-1 text-[11px] font-medium text-slate-400 bg-[#122434] hover:bg-slate-700/60 hover:text-slate-200 border border-white/10 rounded-[6px] transition-all"
                      >
                        OOD Test
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Notice overlay when heatmap is suppressed or unavailable */}
        {!hasHeatmap && viewMode === 'overlay' && originalUrl && Boolean(status) && (
          <div className="absolute bottom-4 left-4 right-4 z-10 flex items-center justify-center pointer-events-none">
            <div className="bg-[#102235]/90 border border-white/10 text-slate-300 text-[11px] px-3.5 py-2 rounded-[8px] backdrop-blur flex items-center gap-2 max-w-md shadow-md">
              <Info className="w-4 h-4 text-teal-400 flex-shrink-0" />
              <span>
                {isRejectedOrAbstained
                  ? 'Grad-CAM overlay is suppressed for rejected or abstained inputs.'
                  : heatmap?.message || 'Grad-CAM activation unavailable for this result.'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Status / Navigation Bar */}
      <div className="px-3.5 py-1.5 bg-[#091522] border-t border-white/10 flex items-center justify-between text-[11px] text-slate-400 flex-wrap gap-2">
        <div className="flex items-center gap-2 truncate">
          {originalUrl ? (
            <>
              <span className="font-medium text-slate-200 truncate max-w-[200px]">
                {fileName || 'Loaded radiograph'}
              </span>
              {fileSize && (
                <span className="font-mono text-slate-400 tabular-nums">
                  &middot; {(fileSize / 1024).toFixed(1)} KB
                </span>
              )}
            </>
          ) : (
            <span>Radiology Canvas &middot; Awaiting input</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {scale > 1 ? (
            <span className="flex items-center gap-1 text-teal-400 font-medium">
              <Move className="w-3.5 h-3.5" />
              <span>Pan enabled</span>
            </span>
          ) : (
            <span className="hidden sm:inline">Fit: 100% &middot; Drag to pan when zoomed</span>
          )}
        </div>
      </div>
    </div>
  );
};
