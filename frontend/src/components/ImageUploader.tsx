import React, { useRef, useState } from 'react';
import { UploadCloud, AlertCircle, ShieldAlert } from 'lucide-react';

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
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [clientError, setClientError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndPass = (file: File) => {
    setClientError(null);

    const isExtensionMatch = /\.(png|jpe?g|webp)$/i.test(file.name);
    const isMimeMatch = ALLOWED_MIME.includes(file.type);
    if (!isExtensionMatch && !isMimeMatch) {
      setClientError('Please select a valid image file (PNG, JPEG, or WEBP).');
      return;
    }

    if (file.size > MAX_BYTES) {
      setClientError('Image size exceeds the 10 MB maximum limit.');
      return;
    }

    if (file.size === 0) {
      setClientError('Selected file is empty (0 bytes).');
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
        aria-label="Upload chest X-ray image for educational decision support"
        onKeyDown={handleKeyDown}
        onClick={triggerSelect}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-[12px] p-8 text-center transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-teal-600 ${
          disabled
            ? 'opacity-50 cursor-not-allowed bg-slate-100 border-border'
            : isDragging
            ? 'border-teal-600 bg-teal-50/50 scale-[0.995]'
            : 'border-border hover:border-teal-600 hover:bg-slate-50/80 bg-surface'
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

        <div className="mx-auto w-12 h-12 rounded-[10px] bg-teal-50 text-teal-700 flex items-center justify-center mb-3.5 border border-teal-100 shadow-2xs">
          <UploadCloud className="w-6 h-6" />
        </div>

        <p className="text-sm font-semibold text-navy-foreground mb-1">
          Drop an educational chest radiograph here, or{' '}
          <span className="text-teal-700 underline underline-offset-2">browse</span>
        </p>
        <p className="text-xs text-navy-muted max-w-sm mx-auto">
          Supported formats: PNG, JPEG, WEBP &middot; Max file size: 10 MB
        </p>

        <div className="mt-4 inline-flex items-center gap-1.5 px-3 py-1 bg-canvas rounded-full text-navy-muted text-[11px] border border-border/80">
          <ShieldAlert className="w-3.5 h-3.5 text-teal-700" />
          <span>Upload only public or de-identified educational images</span>
        </div>
      </div>

      {clientError && (
        <div
          role="alert"
          className="mt-3 p-3 bg-rose-50 border border-rose-200 text-clinical-danger text-xs rounded-[10px] flex items-center gap-2"
        >
          <AlertCircle className="w-4 h-4 flex-shrink-0 text-clinical-danger" />
          <span>{clientError}</span>
        </div>
      )}
    </div>
  );
};
