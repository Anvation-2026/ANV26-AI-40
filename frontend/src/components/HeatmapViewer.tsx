import React, { useState } from 'react';
import { Eye, Layers, Columns, Info, ImageOff } from 'lucide-react';
import { HeatmapInfo } from '../types/analysis';

interface HeatmapViewerProps {
  heatmap: HeatmapInfo;
  originalImageUrl?: string | null;
  status: string;
}

export const HeatmapViewer: React.FC<HeatmapViewerProps> = ({
  heatmap,
  originalImageUrl,
  status,
}) => {
  const [viewMode, setViewMode] = useState<'overlay' | 'heatmap' | 'original' | 'side_by_side'>('overlay');
  const [opacity, setOpacity] = useState<number>(0.6);

  const isRejectedOrAbstained =
    status === 'poor_quality' || status === 'ood' || status === 'uncertain';

  if (!heatmap.available || !heatmap.data_url) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
        <h3 className="text-sm font-bold text-slate-900 tracking-tight mb-3">
          Evidence Localization (Grad-CAM)
        </h3>
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 text-center text-slate-500">
          <ImageOff className="w-8 h-8 mx-auto text-slate-400 mb-2" />
          <p className="text-sm font-medium text-slate-700">
            {isRejectedOrAbstained
              ? 'No heatmap is shown for rejected or abstained inputs.'
              : heatmap.message || 'Heatmap unavailable for this result.'}
          </p>
          <p className="text-xs text-slate-400 mt-1 max-w-md mx-auto">
            Localization overlays are strictly suppressed when image quality is insufficient or model confidence abstains to avoid misleading visual artifacts.
          </p>
        </div>
      </div>
    );
  }

  // Verify safe data URL prefix
  const isSafeDataUrl = heatmap.data_url.startsWith('data:image/png;base64,');
  if (!isSafeDataUrl) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
        <div className="bg-rose-50 border border-rose-200 rounded-xl p-4 text-xs text-rose-800">
          Heatmap data was suppressed due to invalid data format.
        </div>
      </div>
    );
  }

  const isOverlayKind = heatmap.kind === 'overlay';

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900 tracking-tight">
            Evidence Localization (Grad-CAM)
          </h3>
          <span className="text-xs text-slate-400">
            Visual activation overlay influencing model attention
          </span>
        </div>

        {/* View mode toggle tabs */}
        <div className="inline-flex p-1 bg-slate-100 rounded-xl text-xs font-medium self-start sm:self-auto">
          <button
            type="button"
            onClick={() => setViewMode('overlay')}
            className={`px-3 py-1 rounded-lg transition-colors flex items-center gap-1 ${
              viewMode === 'overlay'
                ? 'bg-white text-teal-700 shadow-2xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Overlay</span>
          </button>
          {originalImageUrl && (
            <button
              type="button"
              onClick={() => setViewMode('original')}
              className={`px-3 py-1 rounded-lg transition-colors flex items-center gap-1 ${
                viewMode === 'original'
                  ? 'bg-white text-teal-700 shadow-2xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Original</span>
            </button>
          )}
          {originalImageUrl && (
            <button
              type="button"
              onClick={() => setViewMode('side_by_side')}
              className={`px-3 py-1 rounded-lg transition-colors flex items-center gap-1 ${
                viewMode === 'side_by_side'
                  ? 'bg-white text-teal-700 shadow-2xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Columns className="w-3.5 h-3.5" />
              <span>Side-by-Side</span>
            </button>
          )}
        </div>
      </div>

      {/* Heatmap-only opacity slider if kind is heatmap_only */}
      {!isOverlayKind && viewMode === 'overlay' && (
        <div className="mb-4 flex items-center gap-3 bg-slate-50 p-2.5 rounded-xl border border-slate-200 text-xs">
          <span className="font-medium text-slate-700">Overlay Opacity:</span>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-36 accent-teal-600"
          />
          <span className="text-slate-500 font-mono text-[11px]">
            {Math.round(opacity * 100)}%
          </span>
        </div>
      )}

      {/* Visual display area */}
      <div className="relative bg-slate-950 rounded-xl overflow-hidden p-4 min-h-[280px] flex items-center justify-center">
        {viewMode === 'side_by_side' && originalImageUrl ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
            <div className="text-center">
              <span className="text-[11px] font-semibold text-slate-400 mb-1.5 block uppercase tracking-wider">
                Original Image
              </span>
              <img
                src={originalImageUrl}
                alt="Original educational chest X-ray"
                className="max-h-[300px] mx-auto rounded shadow"
              />
            </div>
            <div className="text-center">
              <span className="text-[11px] font-semibold text-teal-400 mb-1.5 block uppercase tracking-wider">
                Grad-CAM Activation
              </span>
              <img
                src={heatmap.data_url}
                alt="Grad-CAM activation overlay"
                className="max-h-[300px] mx-auto rounded shadow"
              />
            </div>
          </div>
        ) : viewMode === 'original' && originalImageUrl ? (
          <img
            src={originalImageUrl}
            alt="Original chest X-ray"
            className="max-h-[340px] w-auto max-w-full rounded shadow"
          />
        ) : (
          /* Overlay / heatmap view */
          <div className="relative">
            {originalImageUrl && !isOverlayKind && (
              <img
                src={originalImageUrl}
                alt="Original background"
                className="max-h-[340px] w-auto max-w-full rounded"
              />
            )}
            <img
              src={heatmap.data_url}
              alt="Grad-CAM activation overlay"
              style={
                !isOverlayKind && originalImageUrl
                  ? { position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', opacity }
                  : undefined
              }
              className="max-h-[340px] w-auto max-w-full rounded shadow"
            />
          </div>
        )}
      </div>

      {/* Mandatory scientific caution note */}
      <div className="mt-3.5 flex items-start gap-2 text-xs text-slate-500 bg-slate-50 p-3 rounded-xl border border-slate-200">
        <Info className="w-4 h-4 text-teal-600 flex-shrink-0 mt-0.5" />
        <span>
          Highlighted regions influenced the model's output. This does not establish a medically confirmed abnormality or anatomical lesion.
        </span>
      </div>
    </div>
  );
};
