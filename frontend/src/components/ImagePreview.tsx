import React, { useEffect, useState } from 'react';
import { X, FileText } from 'lucide-react';

interface ImagePreviewProps {
  file: File;
  onClear?: () => void;
  disabled?: boolean;
}

export const ImagePreview: React.FC<ImagePreviewProps> = ({
  file,
  onClear,
  disabled = false,
}) => {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    const url = URL.createObjectURL(file);
    setObjectUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const handleImageLoaded = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
  };

  const formatSize = (bytes: number): string => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="bg-surface rounded-[12px] border border-border overflow-hidden shadow-xs">
      <div className="p-3 bg-canvas border-b border-border flex items-center justify-between text-xs text-navy-muted">
        <div className="flex items-center gap-2 truncate pr-2">
          <FileText className="w-4 h-4 text-teal-700 flex-shrink-0" />
          <span className="font-semibold text-navy-foreground truncate" title={file.name}>
            {file.name}
          </span>
          <span className="text-border">&bull;</span>
          <span className="font-mono tabular-nums">{formatSize(file.size)}</span>
          {dimensions && (
            <>
              <span className="text-border">&bull;</span>
              <span className="font-mono tabular-nums">
                {dimensions.width}&times;{dimensions.height} px
              </span>
            </>
          )}
        </div>

        {onClear && (
          <button
            type="button"
            onClick={onClear}
            disabled={disabled}
            aria-label="Remove uploaded image"
            className="p-1 hover:bg-slate-200/80 rounded-[6px] text-navy-muted hover:text-navy-foreground disabled:opacity-50 transition-colors focus:outline-none focus:ring-2 focus:ring-teal-600"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      <div className="relative bg-[#08121C] flex items-center justify-center p-4 max-h-[360px] min-h-[220px]">
        {objectUrl && (
          <img
            src={objectUrl}
            alt="Uploaded chest X-ray preview"
            onLoad={handleImageLoaded}
            className="max-h-[320px] w-auto max-w-full object-contain rounded-[8px] shadow-lg border border-white/5"
          />
        )}
      </div>
    </div>
  );
};
