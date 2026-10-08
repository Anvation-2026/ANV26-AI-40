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
    <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-xs">
      <div className="p-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between text-xs text-slate-600">
        <div className="flex items-center gap-2 truncate pr-2">
          <FileText className="w-4 h-4 text-teal-600 flex-shrink-0" />
          <span className="font-medium text-slate-800 truncate" title={file.name}>
            {file.name}
          </span>
          <span className="text-slate-400">&bull;</span>
          <span>{formatSize(file.size)}</span>
          {dimensions && (
            <>
              <span className="text-slate-400">&bull;</span>
              <span>
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
            aria-label="Remove image"
            className="p-1 hover:bg-slate-200 rounded-md text-slate-500 hover:text-slate-800 disabled:opacity-50 transition-colors focus:outline-none focus:ring-2 focus:ring-teal-500"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      <div className="relative bg-slate-900 flex items-center justify-center p-4 max-h-[380px] min-h-[220px]">
        {objectUrl && (
          <img
            src={objectUrl}
            alt="Uploaded chest X-ray preview"
            onLoad={handleImageLoaded}
            className="max-h-[340px] w-auto max-w-full object-contain rounded shadow"
          />
        )}
      </div>
    </div>
  );
};
