import React, { useRef, useState } from 'react';
import { UploadCloud, FileImage, AlertCircle } from 'lucide-react';

interface ImageUploaderProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

const MAX_BYTES = 10 * 1024 * 1024; // 10 MB
const ALLOWED_MIME = ['image/png', 'image/jpeg', 'image/webp'];

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onFileSelected,
  disabled = false,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [clientError, setClientError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndPass = (file: File) => {
    setClientError(null);

    // Format pre-check
    const isExtensionMatch = /\.(png|jpe?g|webp)$/i.test(file.name);
    const isMimeMatch = ALLOWED_MIME.includes(file.type);
    if (!isExtensionMatch && !isMimeMatch) {
      setClientError('Please select a valid image file (PNG, JPEG, or WEBP).');
      return;
    }

    // Size pre-check
    if (file.size > MAX_BYTES) {
      setClientError('Image size exceeds the 10 MB maximum limit.');
      return;
    }

    if (file.size === 0) {
      setClientError('Selected file is empty.');
      return;
    }

    onFileSelected(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (disabled) return;
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndPass(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndPass(e.target.files[0]);
    }
  };

  const triggerSelect = () => {
    if (disabled) return;
    fileInputRef.current?.click();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
      e.preventDefault();
      triggerSelect();
    }
  };

  return (
    <div className="w-full">
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload chest X-ray image for educational analysis"
        onKeyDown={handleKeyDown}
        onClick={triggerSelect}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer focus:outline-none focus:ring-4 focus:ring-teal-500/20 ${
          disabled
            ? 'opacity-50 cursor-not-allowed bg-slate-100 border-slate-300'
            : isDragging
            ? 'border-teal-500 bg-teal-50/70 scale-[0.99]'
            : 'border-slate-300 hover:border-teal-500 hover:bg-slate-50/80 bg-white'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
          onChange={handleFileInputChange}
          disabled={disabled}
          className="sr-only"
          aria-hidden="true"
        />

        <div className="mx-auto w-14 h-14 rounded-2xl bg-teal-50 text-teal-600 flex items-center justify-center mb-4 border border-teal-100 shadow-2xs">
          <UploadCloud className="w-7 h-7" />
        </div>

        <p className="text-base font-semibold text-slate-800 mb-1">
          Drop an educational chest X-ray here, or{' '}
          <span className="text-teal-600 underline underline-offset-2">browse</span>
        </p>
        <p className="text-xs text-slate-500 max-w-sm mx-auto">
          Supported formats: PNG, JPEG, WEBP &middot; Max file size: 10 MB
        </p>

        <div className="mt-4 inline-flex items-center gap-1.5 px-3 py-1 bg-slate-100 rounded-full text-slate-600 text-xs">
          <FileImage className="w-3.5 h-3.5" />
          <span>Use public or de-identified educational images only</span>
        </div>
      </div>

      {clientError && (
        <div
          role="alert"
          className="mt-3 p-3 bg-rose-50 border border-rose-200 text-rose-800 text-xs rounded-xl flex items-center gap-2"
        >
          <AlertCircle className="w-4 h-4 flex-shrink-0 text-rose-600" />
          <span>{clientError}</span>
        </div>
      )}
    </div>
  );
};
