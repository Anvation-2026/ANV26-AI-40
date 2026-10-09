import React, { useState } from 'react';
import { Menu, ShieldAlert, Cpu, Server, X, CheckCircle2, AlertCircle } from 'lucide-react';
import { getApiBase, setApiBase } from '../services/api';

interface TopBarProps {
  onToggleMobileNav: () => void;
  pageTitle: string;
  backendAlive: boolean | null;
  backendMode?: 'real' | 'demo';
}

export const TopBar: React.FC<TopBarProps> = ({
  onToggleMobileNav,
  pageTitle,
  backendAlive,
  backendMode = 'demo',
}) => {
  const [showModal, setShowModal] = useState(false);
  const [customUrl, setCustomUrl] = useState(() => getApiBase());
  const [savedSuccess, setSavedSuccess] = useState(false);

  const handleSaveUrl = (e: React.FormEvent) => {
    e.preventDefault();
    setApiBase(customUrl.trim());
    setSavedSuccess(true);
    setTimeout(() => {
      window.location.reload();
    }, 500);
  };

  const handleResetToDemo = () => {
    setApiBase('');
    setCustomUrl('');
    setSavedSuccess(true);
    setTimeout(() => {
      window.location.reload();
    }, 500);
  };

  return (
    <header className="bg-surface border-b border-border sticky top-0 z-20 flex-shrink-0">
      {/* Persistent Disclaimer Ribbon */}
      <aside aria-label="Educational Disclaimer" className="bg-[#FFFBEB] text-[#92400E] px-4 py-1.5 text-[11px] font-medium border-b border-amber-200/60 flex items-center justify-center gap-2">
        <ShieldAlert className="w-3.5 h-3.5 text-clinical-warning flex-shrink-0" />
        <span className="text-center truncate">
          Educational research prototype. Not a medical device. Not for diagnosis or treatment decisions.
        </span>
      </aside>

      {/* Main Top Bar */}
      <div className="h-14 px-4 sm:px-6 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onToggleMobileNav}
            className="md:hidden p-1.5 rounded-[8px] hover:bg-slate-100 text-navy-muted hover:text-navy focus:outline-none focus:ring-2 focus:ring-teal-600"
            aria-label="Toggle navigation"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div>
            <h1 className="text-sm sm:text-base font-bold text-navy-foreground tracking-tight">
              {pageTitle}
            </h1>
          </div>
        </div>

        {/* Right side status items */}
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-100/80 border border-slate-200/80 text-[11px] text-navy-muted">
            <span className="font-medium text-navy-foreground">Dataset:</span>
            <span>PneumoniaMNIST+ (224&times;224)</span>
          </div>

          <button
            type="button"
            onClick={() => setShowModal(true)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border cursor-pointer hover:shadow-xs transition-colors"
            title="Click to configure backend connection"
          >
            {backendMode === 'real' && backendAlive ? (
              <>
                <span className="w-2 h-2 rounded-full bg-clinical-success animate-pulse"></span>
                <span className="text-clinical-success text-[11px] font-semibold">PyTorch Online</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                <span className="text-amber-700 text-[11px] font-semibold">Demo Mode</span>
              </>
            )}
            <Server className="w-3 h-3 text-slate-400 ml-0.5" />
          </button>
        </div>
      </div>

      {/* Backend Settings Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-navy/60 backdrop-blur-xs animate-in fade-in">
          <div className="bg-white rounded-2xl shadow-xl max-w-lg w-full p-6 border border-slate-200">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2">
                <Cpu className="w-5 h-5 text-teal-600" />
                <h3 className="text-base font-bold text-navy">AI Model &amp; Backend Connection</h3>
              </div>
              <button
                type="button"
                onClick={() => setShowModal(false)}
                className="p-1 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="mt-4 space-y-4">
              <div className={`p-3 rounded-xl border text-xs flex items-start gap-2.5 ${backendMode === 'real' ? 'bg-emerald-50 border-emerald-200 text-emerald-900' : 'bg-amber-50 border-amber-200 text-amber-900'}`}>
                {backendMode === 'real' ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 mt-0.5 flex-shrink-0" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                )}
                <div>
                  <p className="font-semibold">
                    {backendMode === 'real' ? 'Connected to Live Python Backend' : 'Running in Client-Side Interactive Demo'}
                  </p>
                  <p className="mt-0.5 opacity-90 leading-relaxed">
                    {backendMode === 'real'
                      ? 'The frontend is streaming inferences and Grad-CAM directly from your PyTorch neural network.'
                      : 'GitHub Pages & static web CDNs only serve static files (HTML/JS) and cannot run Python or PyTorch deep learning models directly on GitHub servers.'}
                  </p>
                </div>
              </div>

              <div className="text-xs space-y-2 text-slate-600">
                <p className="font-semibold text-slate-800">To run the Real PyTorch Model:</p>
                <ol className="list-decimal pl-4 space-y-1">
                  <li>
                    <strong>Locally:</strong> Run <code className="bg-slate-100 px-1 py-0.5 rounded text-teal-700">.\start.ps1</code> or <code className="bg-slate-100 px-1 py-0.5 rounded text-teal-700">python -m uvicorn backend.src.main:app --port 8000</code>.
                  </li>
                  <li>
                    <strong>Cloud Backend:</strong> Deploy the <code className="bg-slate-100 px-1 py-0.5 rounded">backend/</code> container to Render, Railway, or Hugging Face Spaces.
                  </li>
                </ol>
              </div>

              <form onSubmit={handleSaveUrl} className="pt-2 border-t border-slate-100">
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Custom Backend API Base URL:
                </label>
                <div className="flex gap-2">
                  <input
                    type="url"
                    placeholder="e.g. http://localhost:8000 or https://api.my-domain.com"
                    value={customUrl}
                    onChange={(e) => setCustomUrl(e.target.value)}
                    className="flex-1 text-xs px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 font-mono"
                  />
                  <button
                    type="submit"
                    className="px-3 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-lg text-xs font-semibold whitespace-nowrap"
                  >
                    Save &amp; Connect
                  </button>
                </div>
                {customUrl && (
                  <button
                    type="button"
                    onClick={handleResetToDemo}
                    className="mt-2 text-[11px] text-slate-500 hover:text-slate-800 underline"
                  >
                    Reset to Default Demo Mode
                  </button>
                )}
                {savedSuccess && (
                  <p className="text-xs text-emerald-600 mt-2 font-medium">
                    Settings saved! Reloading...
                  </p>
                )}
              </form>
            </div>
          </div>
        </div>
      )}
    </header>
  );
};
