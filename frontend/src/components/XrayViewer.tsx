import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Layers,
  Columns,
  Eye,
  Info,
  ImageOff,
  Maximize2,
  Minimize2,
  Move,
} from 'lucide-react';
import { HeatmapInfo } from '../types/analysis';

interface XrayViewerProps {
  originalUrl: string | null;
  heatmap: HeatmapInfo | null;
  status?: string;
  className?: string;
}

export const XrayViewer: React.FC<XrayViewerProps> = ({
  originalUrl,
  heatmap,
  status,
  className = '',
}) => {
  const [scale, setScale] = useState<number>(1);
  const [position, setPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [viewMode, setViewMode] = useState<'overlay' | 'original' | 'side_by_side' | 'heatmap_only'>('overlay');
  const [opacity, setOpacity] = useState<number>(0.65);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  const containerRef = useRef<HTMLDivElement>(null);

  const hasHeatmap = Boolean(heatmap?.available && heatmap?.data_url?.startsWith('data:image/png;base64,'));
  const isRejectedOrAbstained = status === 'poor_quality' || status === 'ood' || status === 'uncertain';

  const resetView = useCallback(() => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  }, []);

  const handleZoom = (delta: number) => {
    setScale((prev) => {
      const next = Math.min(Math.max(prev + delta, 0.75), 4.0);
      if (next === 1) setPosition({ x: 0, y: 0 });
      return next;
    });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (scale <= 1) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || scale <= 1) return;
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      handleZoom(e.deltaY < 0 ? 0.2 : -0.2);
    }
  };

  useEffect(() => {
    resetView();
  }, [originalUrl, heatmap?.data_url, resetView]);

  return (
    <div
      ref={containerRef}
      className={`relative flex flex-col bg-[#0B1520] border border-border/80 rounded-[12px] overflow-hidden select-none transition-all shadow-sm ${
        isFullscreen ? 'fixed inset-4 z-50 rounded-xl' : 'w-full min-h-[460px] max-h-[640px]'
      } ${className}`}
      onWheel={handleWheel}
    >
      {/* Top Clinical Toolbar */}
      <div className="flex items-center justify-between px-3.5 py-2.5 bg-[#102235]/95 border-b border-border/10 backdrop-blur z-20 text-white text-xs">
        {/* Left: View Mode Toggles */}
        <div className="flex items-center gap-1.5 bg-[#08121C] p-1 rounded-[8px] border border-white/5">
          <button
            type="button"
            onClick={() => setViewMode('overlay')}
            className={`px-2.5 py-1 rounded-[6px] font-medium transition-colors flex items-center gap-1.5 ${
              viewMode === 'overlay' ? 'bg-teal-700 text-white shadow-xs' : 'text-slate-300 hover:text-white'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Overlay</span>
          </button>

          {originalUrl && (
            <button
              type="button"
              onClick={() => setViewMode('original')}
              className={`px-2.5 py-1 rounded-[6px] font-medium transition-colors flex items-center gap-1.5 ${
                viewMode === 'original' ? 'bg-teal-700 text-white shadow-xs' : 'text-slate-300 hover:text-white'
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Original</span>
            </button>
          )}

          {hasHeatmap && originalUrl && (
            <button
              type="button"
              onClick={() => setViewMode('side_by_side')}
              className={`px-2.5 py-1 rounded-[6px] font-medium transition-colors flex items-center gap-1.5 ${
                viewMode === 'side_by_side' ? 'bg-teal-700 text-white shadow-xs' : 'text-slate-300 hover:text-white'
              }`}
            >
              <Columns className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Side-by-Side</span>
            </button>
          )}
        </div>

        {/* Center: Opacity Slider (when overlay mode and heatmap available) */}
        {hasHeatmap && viewMode === 'overlay' && (
          <div className="hidden md:flex items-center gap-2 bg-[#08121C] px-3 py-1 rounded-[8px] border border-white/5">
            <span className="text-[11px] text-slate-300 font-medium">Grad-CAM Opacity</span>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-20 accent-teal-500 cursor-pointer h-1.5 bg-slate-700 rounded-lg"
            />
            <span className="text-[11px] font-mono tabular-nums text-teal-400 w-8 text-right">
              {Math.round(opacity * 100)}%
            </span>
          </div>
        )}

        {/* Right: Inspection Controls (Zoom, Pan, Fit, Fullscreen) */}
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => handleZoom(-0.25)}
            disabled={scale <= 0.75}
            title="Zoom Out (Ctrl -)"
            aria-label="Zoom Out"
            className="p-1.5 rounded-[6px] hover:bg-white/10 text-slate-300 disabled:opacity-30 transition-colors"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-[11px] font-mono tabular-nums px-1 text-slate-300 min-w-[38px] text-center">
            {Math.round(scale * 100)}%
          </span>
          <button
            type="button"
            onClick={() => handleZoom(0.25)}
            disabled={scale >= 4.0}
            title="Zoom In (Ctrl +)"
            aria-label="Zoom In"
            className="p-1.5 rounded-[6px] hover:bg-white/10 text-slate-300 disabled:opacity-30 transition-colors"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={resetView}
            title="Reset View (Fit to window)"
            aria-label="Reset View"
            className="p-1.5 rounded-[6px] hover:bg-white/10 text-slate-300 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <div className="h-4 w-px bg-white/10 mx-0.5"></div>
          <button
            type="button"
            onClick={() => setIsFullscreen(!isFullscreen)}
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen Viewer'}
            aria-label={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen Viewer'}
            className="p-1.5 rounded-[6px] hover:bg-white/10 text-slate-300 transition-colors"
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Main Canvas Viewport */}
      <div
        className="flex-1 relative overflow-hidden flex items-center justify-center p-4 cursor-default"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: scale > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default' }}
      >
        {/* Subtle grid pattern background */}
        <div
          className="absolute inset-0 opacity-[0.03] pointer-events-none"
          style={{
            backgroundImage: 'radial-gradient(circle, #ffffff 1px, transparent 1px)',
            backgroundSize: '24px 24px',
          }}
        />

        {/* Side-by-side mode */}
        {viewMode === 'side_by_side' && originalUrl && hasHeatmap ? (
          <div
            className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-4xl transition-transform"
            style={{
              transform: `scale(${scale}) translate(${position.x / scale}px, ${position.y / scale}px)`,
              transformOrigin: 'center center',
            }}
          >
            <div className="flex flex-col items-center">
              <span className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase mb-2">
                Original Radiograph
              </span>
              <img
                src={originalUrl}
                alt="Original educational chest X-ray"
                className="max-h-[380px] w-auto max-w-full rounded-[8px] shadow-lg border border-white/10 object-contain pointer-events-none"
              />
            </div>
            <div className="flex flex-col items-center">
              <span className="text-[11px] font-semibold tracking-wider text-teal-400 uppercase mb-2">
                Grad-CAM Activation
              </span>
              <img
                src={heatmap?.data_url || ''}
                alt="Grad-CAM activation heatmap"
                className="max-h-[380px] w-auto max-w-full rounded-[8px] shadow-lg border border-teal-500/20 object-contain pointer-events-none"
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
                {/* Base radiograph */}
                <img
                  src={originalUrl}
                  alt="Chest radiograph preview"
                  className="max-h-[440px] w-auto max-w-full rounded-[8px] shadow-2xl object-contain border border-white/5 pointer-events-none"
                />

                {/* Grad-CAM overlay */}
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
              <div className="text-center text-slate-500 p-8">
                <ImageOff className="w-10 h-10 mx-auto text-slate-600 mb-2" />
                <p className="text-xs font-medium">No image loaded into viewer</p>
              </div>
            )}
          </div>
        )}

        {/* Notice overlay when heatmap is suppressed or unavailable */}
        {!hasHeatmap && viewMode === 'overlay' && originalUrl && (
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
      <div className="px-3.5 py-2 bg-[#102235] border-t border-border/10 flex items-center justify-between text-[11px] text-slate-400">
        <div className="flex items-center gap-2">
          {scale > 1 ? (
            <span className="flex items-center gap-1 text-teal-400 font-medium">
              <Move className="w-3.5 h-3.5" />
              <span>Pan enabled (drag image)</span>
            </span>
          ) : (
            <span>Fit: 100% &middot; Drag to pan when zoomed</span>
          )}
        </div>
        <div className="truncate max-w-xs text-right">
          Highlighted regions reflect feature importance &middot; Not confirmed pathology
        </div>
      </div>
    </div>
  );
};
