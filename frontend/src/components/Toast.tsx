import React, { createContext, useContext, useState, useCallback } from 'react';
import { AlertCircle, CheckCircle2, Info, X } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'info';

interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

interface ToastContextValue {
  showToast: (message: string, type?: ToastType) => void;
}

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const showToast = useCallback((message: string, type: ToastType = 'info') => {
    const id = Math.random().toString(36).substring(2, 9);
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4500);
  }, []);

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      <div
        aria-live="polite"
        className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none"
      >
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`pointer-events-auto flex items-start gap-3 p-3.5 rounded-[12px] border shadow-md text-xs font-medium transition-all transform duration-200 ${
              toast.type === 'error'
                ? 'bg-white border-clinical-danger/30 text-clinical-danger'
                : toast.type === 'success'
                ? 'bg-white border-clinical-success/30 text-clinical-success'
                : 'bg-white border-border text-navy-foreground'
            }`}
          >
            {toast.type === 'error' && <AlertCircle className="w-4 h-4 flex-shrink-0 text-clinical-danger mt-0.5" />}
            {toast.type === 'success' && <CheckCircle2 className="w-4 h-4 flex-shrink-0 text-clinical-success mt-0.5" />}
            {toast.type === 'info' && <Info className="w-4 h-4 flex-shrink-0 text-teal-700 mt-0.5" />}
            <span className="flex-1 text-navy-foreground leading-relaxed">{toast.message}</span>
            <button
              onClick={() => removeToast(toast.id)}
              className="text-navy-muted hover:text-navy p-0.5 rounded transition-colors"
              aria-label="Dismiss notification"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
};

export const useToast = (): ToastContextValue => {
  const context = useContext(ToastContext);
  if (!context) {
    return {
      showToast: () => {},
    };
  }
  return context;
};
