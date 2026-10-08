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

  let maxVal = 0;
  for (let r = 0; r < numRows; r++) {
    for (let c = 0; c < numCols; c++) {
      if (matrix[r][c] > maxVal) maxVal = matrix[r][c];
    }
  }

  const getIntensityClass = (val: number, isDiagonal: boolean) => {
    if (val === 0) return 'bg-canvas text-navy-muted border-border';
    if (!maxVal) return 'bg-slate-100 text-navy-foreground border-border';

    const ratio = val / maxVal;
    if (isDiagonal) {
      if (ratio > 0.6) return 'bg-teal-700 text-white font-bold border-teal-800';
      if (ratio > 0.3) return 'bg-teal-600 text-white font-semibold border-teal-700';
      return 'bg-teal-100 text-teal-950 font-medium border-teal-200';
    } else {
      if (ratio > 0.3) return 'bg-rose-600 text-white font-bold border-rose-700';
      if (ratio > 0.1) return 'bg-rose-200 text-rose-950 font-semibold border-rose-300';
      return 'bg-rose-50 text-rose-900 font-medium border-rose-200';
    }
  };

  return (
    <div className="bg-surface rounded-[12px] border border-border p-5 shadow-xs">
      <div className="mb-4">
        <h3 className="text-sm font-bold text-navy-foreground tracking-tight">
          Confusion Matrix (Test Benchmark)
        </h3>
        <span className="text-xs text-navy-muted">
          Predicted vs. Ground-Truth Classes on PneumoniaMNIST+ Partition
        </span>
      </div>

      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          {/* Header Row: Predicted Labels */}
          <div className="grid grid-cols-[110px_repeat(2,minmax(130px,1fr))] gap-2.5 mb-2.5">
            <div></div>
            {labels.map((lbl, idx) => (
              <div
                key={idx}
                className="text-center text-xs font-semibold uppercase tracking-wider text-navy-muted"
              >
                Pred: {lbl}
              </div>
            ))}
          </div>

          {/* Matrix Rows */}
          {matrix.map((row, rowIdx) => (
            <div
              key={rowIdx}
              className="grid grid-cols-[110px_repeat(2,minmax(130px,1fr))] gap-2.5 mb-2.5 items-center"
            >
              <div className="text-right pr-3 text-xs font-semibold uppercase tracking-wider text-navy-muted">
                True: {labels[rowIdx] || `Class ${rowIdx}`}
              </div>
              {row.map((val, colIdx) => {
                const isDiagonal = rowIdx === colIdx;
                const cellClass = getIntensityClass(val, isDiagonal);
                return (
                  <div
                    key={colIdx}
                    className={`h-16 rounded-[9px] flex flex-col items-center justify-center border shadow-2xs transition-transform hover:scale-[1.01] ${cellClass}`}
                  >
                    <span className="text-lg font-mono tabular-nums font-bold">{val}</span>
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
