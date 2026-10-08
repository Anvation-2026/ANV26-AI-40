import React from 'react';
import { ConfusionMatrixData } from '../types/analysis';

interface ConfusionMatrixProps {
  data: ConfusionMatrixData | null | undefined;
}

export const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({ data }) => {
  if (!data || !data.matrix || data.matrix.length === 0) {
    return null;
  }

  const { labels, matrix } = data;
  const numRows = matrix.length;
  const numCols = matrix[0]?.length || 0;

  // Compute maximum cell value for heat intensity
  let maxVal = 0;
  for (let r = 0; r < numRows; r++) {
    for (let c = 0; c < numCols; c++) {
      if (matrix[r][c] > maxVal) maxVal = matrix[r][c];
    }
  }

  const getIntensityClass = (val: number, isDiagonal: boolean) => {
    if (val === 0) return 'bg-slate-50 text-slate-400';
    if (!maxVal) return 'bg-slate-100 text-slate-700';

    const ratio = val / maxVal;
    if (isDiagonal) {
      if (ratio > 0.6) return 'bg-emerald-600 text-white font-bold';
      if (ratio > 0.3) return 'bg-emerald-400 text-slate-900 font-semibold';
      return 'bg-emerald-100 text-emerald-950 font-medium';
    } else {
      if (ratio > 0.3) return 'bg-rose-500 text-white font-bold';
      if (ratio > 0.1) return 'bg-rose-200 text-rose-950 font-semibold';
      return 'bg-rose-50 text-rose-900 font-medium';
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
      <div className="mb-4">
        <h3 className="text-sm font-bold text-slate-900 tracking-tight">
          Confusion Matrix
        </h3>
        <span className="text-xs text-slate-400">
          Predicted vs True Ground Truth Classes on Test Partition
        </span>
      </div>

      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          {/* Header Row: Predicted Labels */}
          <div className="grid grid-cols-[100px_repeat(2,minmax(120px,1fr))] gap-2 mb-2">
            <div></div>
            {labels.map((lbl, idx) => (
              <div
                key={idx}
                className="text-center text-xs font-semibold uppercase tracking-wider text-slate-500"
              >
                Pred: {lbl}
              </div>
            ))}
          </div>

          {/* Matrix Rows */}
          {matrix.map((row, rowIdx) => (
            <div
              key={rowIdx}
              className="grid grid-cols-[100px_repeat(2,minmax(120px,1fr))] gap-2 mb-2 items-center"
            >
              <div className="text-right pr-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
                True: {labels[rowIdx] || `C${rowIdx}`}
              </div>
              {row.map((val, colIdx) => {
                const isDiagonal = rowIdx === colIdx;
                const cellClass = getIntensityClass(val, isDiagonal);
                return (
                  <div
                    key={colIdx}
                    className={`h-16 rounded-xl flex flex-col items-center justify-center border border-slate-200/80 shadow-2xs transition-transform hover:scale-[1.02] ${cellClass}`}
                  >
                    <span className="text-lg font-mono font-bold">{val}</span>
                    <span className="text-[10px] opacity-80 uppercase tracking-wider">
                      {isDiagonal ? 'Correct' : 'Error'}
                    </span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
