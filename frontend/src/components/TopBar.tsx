import React from 'react';
import { Menu, ShieldAlert } from 'lucide-react';

interface TopBarProps {
  onToggleMobileNav: () => void;
  pageTitle: string;
  backendAlive: boolean | null;
}

export const TopBar: React.FC<TopBarProps> = ({
  onToggleMobileNav,
  pageTitle,
  backendAlive,
}) => {
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

          <div
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border"
            title={backendAlive ? 'API Connected (:8000)' : 'API Unreachable'}
          >
            {backendAlive === true ? (
              <>
                <span className="w-2 h-2 rounded-full bg-clinical-success animate-pulse"></span>
                <span className="text-clinical-success text-[11px] font-semibold">API Online</span>
              </>
            ) : backendAlive === false ? (
              <>
                <span className="w-2 h-2 rounded-full bg-clinical-danger"></span>
                <span className="text-clinical-danger text-[11px] font-semibold">API Offline</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-slate-400"></span>
                <span className="text-navy-muted text-[11px]">Connecting</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
