import React, { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  ShieldAlert,
  Activity,
  BarChart2,
  FileQuestion,
  Search,
} from 'lucide-react';
import { getHealth } from '../services/api';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();
  const [backendAlive, setBackendAlive] = useState<boolean | null>(null);

  useEffect(() => {
    let mounted = true;
    getHealth()
      .then(() => {
        if (mounted) setBackendAlive(true);
      })
      .catch(() => {
        if (mounted) setBackendAlive(false);
      });
    return () => {
      mounted = false;
    };
  }, [location.pathname]);

  const navLinks = [
    { to: '/', label: 'Dashboard', icon: Activity },
    { to: '/analyze', label: 'Analyze', icon: Search },
    { to: '/validation', label: 'Validation', icon: BarChart2 },
    { to: '/about', label: 'Methodology', icon: FileQuestion },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
      {/* Persistent Educational Notice Banner */}
      <aside aria-label="Educational Disclaimer" className="bg-amber-500 text-amber-950 px-4 py-2 text-xs font-semibold tracking-wide border-b border-amber-600 shadow-sm flex items-center justify-center gap-2">
        <ShieldAlert className="w-4 h-4 text-amber-950 flex-shrink-0" />
        <span className="text-center">
          Educational research prototype. Not a medical device. Not for diagnosis or treatment decisions. Use only public or de-identified educational images.
        </span>
      </aside>

      {/* Main Navigation */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <Link to="/" className="flex items-center gap-2.5 focus:outline-none focus:ring-2 focus:ring-teal-500 rounded-lg p-1">
              <div className="w-10 h-10 rounded-xl bg-teal-600 text-white flex items-center justify-center font-bold text-xl shadow-sm">
                M
              </div>
              <div>
                <span className="text-lg font-bold text-slate-900 tracking-tight">MedGuard <span className="text-teal-600 font-extrabold">AI</span></span>
                <span className="hidden sm:inline-block ml-2 px-2 py-0.5 text-[11px] font-medium bg-slate-100 text-slate-600 rounded-full border border-slate-200">
                  Triage Decision Support
                </span>
              </div>
            </Link>

            {/* Navigation Links */}
            <nav className="flex items-center gap-1 sm:gap-2">
              {navLinks.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.to;
                return (
                  <Link
                    key={item.to}
                    to={item.to}
                    className={`flex items-center gap-1.5 px-3 py-2 text-sm font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-teal-500 ${
                      isActive
                        ? 'bg-teal-50 text-teal-700 font-semibold border border-teal-200/60'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                    }`}
                  >
                    <Icon className={`w-4 h-4 ${isActive ? 'text-teal-600' : 'text-slate-500'}`} />
                    <span>{item.label}</span>
                  </Link>
                );
              })}

              {/* Backend status indicator */}
              <div className="ml-2 pl-3 border-l border-slate-200 hidden md:flex items-center gap-1.5 text-xs font-medium text-slate-500" title={backendAlive ? 'API Connected' : 'API Offline'}>
                {backendAlive === true ? (
                  <>
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    <span className="text-slate-600">API Live</span>
                  </>
                ) : backendAlive === false ? (
                  <>
                    <span className="w-2 h-2 rounded-full bg-red-500"></span>
                    <span className="text-red-600">API Offline</span>
                  </>
                ) : (
                  <>
                    <span className="w-2 h-2 rounded-full bg-slate-300"></span>
                    <span className="text-slate-400">Connecting</span>
                  </>
                )}
              </div>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <p>
            MedGuard AI &middot; Educational Decision-Support Research System &middot; Team Vouken
          </p>
          <p className="text-slate-400 text-center md:text-right">
            Strictly for instructional & demonstration purposes &middot; Not for medical diagnostics
          </p>
        </div>
      </footer>
    </div>
  );
};
