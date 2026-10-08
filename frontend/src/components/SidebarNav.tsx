import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Activity,
  Scan,
  BarChart3,
  BookOpen,
  Cpu,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  FlaskConical,
} from 'lucide-react';
import { ModelStatusResponse } from '../types/analysis';

interface SidebarNavProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  modelStatus: ModelStatusResponse | null;
}

export const SidebarNav: React.FC<SidebarNavProps> = ({
  collapsed,
  onToggleCollapse,
  modelStatus,
}) => {
  const location = useLocation();

  const navItems = [
    { to: '/', label: 'Overview', icon: Activity },
    { to: '/analyze', label: 'Analysis Workspace', icon: Scan },
    { to: '/validation', label: 'Model Validation', icon: BarChart3 },
    { to: '/about', label: 'Methodology & Trust', icon: BookOpen },
  ];

  return (
    <aside
      className={`bg-navy text-white transition-all duration-200 ease-in-out flex flex-col justify-between border-r border-border/10 flex-shrink-0 z-30 ${
        collapsed ? 'w-16' : 'w-60'
      }`}
    >
      {/* Brand Header */}
      <div>
        <div className="h-16 flex items-center justify-between px-3.5 border-b border-white/10">
          <Link
            to="/"
            className="flex items-center gap-2.5 overflow-hidden focus:outline-none focus:ring-2 focus:ring-teal-500 rounded-lg p-1"
          >
            <div className="w-8 h-8 rounded-[8px] bg-teal-700 text-white flex items-center justify-center font-bold text-base shadow-xs flex-shrink-0">
              M
            </div>
            {!collapsed && (
              <div className="truncate">
                <span className="text-sm font-bold tracking-tight text-white">
                  MedGuard <span className="text-teal-400 font-extrabold">AI</span>
                </span>
                <span className="block text-[10px] text-slate-300 font-medium tracking-wide uppercase">
                  Clinical Triage
                </span>
              </div>
            )}
          </Link>

          <button
            type="button"
            onClick={onToggleCollapse}
            className="p-1 rounded-[6px] text-slate-300 hover:text-white hover:bg-white/10 transition-colors hidden md:flex items-center justify-center"
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Navigation Links */}
        <nav className="p-2 space-y-1 mt-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.to;
            return (
              <Link
                key={item.to}
                to={item.to}
                title={collapsed ? item.label : undefined}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-[9px] text-xs font-medium transition-colors ${
                  isActive
                    ? 'bg-teal-700/90 text-white font-semibold shadow-xs'
                    : 'text-slate-200 hover:bg-white/5 hover:text-white'
                }`}
              >
                <Icon className={`w-4 h-4 flex-shrink-0 ${isActive ? 'text-white' : 'text-slate-300'}`} />
                {!collapsed && <span className="truncate">{item.label}</span>}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Footer System Status Panel */}
      <div className="p-3 border-t border-white/10 bg-[#0E1E2E]">
        {!collapsed ? (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-[11px] text-slate-300">
              <span className="flex items-center gap-1.5 font-medium">
                <Cpu className="w-3.5 h-3.5 text-teal-400" />
                <span>Engine Status</span>
              </span>
              {modelStatus?.mode === 'demo' ? (
                <span className="inline-flex items-center gap-1 text-amber-400 font-medium">
                  <FlaskConical className="w-3 h-3" />
                  <span>Demo</span>
                </span>
              ) : modelStatus?.inference_ready ? (
                <span className="inline-flex items-center gap-1 text-emerald-400 font-medium">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                  <span>Active</span>
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 text-slate-400 font-medium">
                  <span className="w-1.5 h-1.5 rounded-full bg-slate-500"></span>
                  <span>Offline</span>
                </span>
              )}
            </div>

            <div className="p-2 rounded-[8px] bg-white/5 border border-white/5 text-[10px] text-slate-300 space-y-0.5">
              <div className="flex justify-between">
                <span>Model:</span>
                <span className="font-mono text-slate-200 truncate max-w-[100px]">
                  {modelStatus?.model_name || 'Unavailable'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Domain:</span>
                <span className="text-slate-200">Chest X-Ray</span>
              </div>
            </div>

            <div className="pt-1 flex items-center gap-1 text-[9px] text-slate-400">
              <ShieldCheck className="w-3 h-3 text-teal-400 flex-shrink-0" />
              <span className="truncate">PneumoniaMNIST+ Research</span>
            </div>
          </div>
        ) : (
          <div className="flex justify-center" title="Model Status">
            {modelStatus?.mode === 'demo' ? (
              <FlaskConical className="w-4 h-4 text-amber-400" />
            ) : modelStatus?.inference_ready ? (
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
            ) : (
              <span className="w-2.5 h-2.5 rounded-full bg-slate-500"></span>
            )}
          </div>
        )}
      </div>
    </aside>
  );
};
