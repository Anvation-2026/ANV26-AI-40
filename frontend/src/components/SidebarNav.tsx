import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Scan,
  BarChart3,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  Radio,
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
    { to: '/', label: 'Workspace', icon: Scan },
    { to: '/validation', label: 'Validation', icon: BarChart3 },
    { to: '/about', label: 'About & Methodology', icon: BookOpen },
  ];

  return (
    <aside
      className={`bg-[#0d1d2b] text-white transition-all duration-200 ease-in-out flex flex-col justify-between border-r border-white/10 flex-shrink-0 z-30 select-none ${
        collapsed ? 'w-16' : 'w-[220px]'
      }`}
    >
      {/* Brand Header */}
      <div>
        <div className="h-14 flex items-center justify-between px-3 border-b border-white/10">
          <Link
            to="/"
            className="flex items-center gap-2.5 overflow-hidden focus:outline-none focus:ring-2 focus:ring-teal-500 rounded-lg p-1 group"
          >
            <div className="w-7 h-7 rounded-[7px] bg-teal-600 text-white flex items-center justify-center font-bold text-sm shadow-xs flex-shrink-0 group-hover:bg-teal-500 transition-colors">
              M
            </div>
            {!collapsed && (
              <div className="truncate">
                <div className="text-xs font-bold tracking-tight text-white flex items-center gap-1">
                  <span>MedGuard</span>
                  <span className="text-teal-400 font-extrabold text-[11px] px-1 py-0.2 bg-teal-950/80 rounded border border-teal-500/40">
                    AI
                  </span>
                </div>
                <span className="block text-[9px] text-slate-400 font-medium tracking-wider uppercase">
                  Radiology Triage
                </span>
              </div>
            )}
          </Link>

          <button
            type="button"
            onClick={onToggleCollapse}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="p-1.5 rounded-[6px] hover:bg-white/10 text-slate-400 hover:text-white transition-colors focus:outline-none focus:ring-1 focus:ring-teal-500"
          >
            {collapsed ? <ChevronRight className="w-3.5 h-3.5" /> : <ChevronLeft className="w-3.5 h-3.5" />}
          </button>
        </div>

        {/* Navigation Destination Links */}
        <nav className="p-2 space-y-1 mt-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive =
              item.to === '/'
                ? location.pathname === '/' || location.pathname === '/analyze'
                : location.pathname.startsWith(item.to);

            return (
              <Link
                key={item.to}
                to={item.to}
                className="relative block"
                title={collapsed ? item.label : undefined}
              >
                <div
                  className={`relative z-10 flex items-center gap-3 px-3 py-2.5 rounded-[8px] text-xs font-medium transition-colors ${
                    isActive
                      ? 'text-white font-semibold'
                      : 'text-slate-300 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Icon className={`w-4 h-4 flex-shrink-0 ${isActive ? 'text-teal-400' : 'text-slate-400'}`} />
                  {!collapsed && <span className="truncate">{item.label}</span>}
                </div>

                {isActive && (
                  <motion.div
                    layoutId="active-nav-indicator"
                    className="absolute inset-0 bg-teal-800/80 border border-teal-500/30 rounded-[8px] shadow-xs z-0"
                    transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                  />
                )}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Model Engine Status Card Footer */}
      <div className="p-2.5 border-t border-white/10 bg-[#07131e]/90 text-xs">
        {collapsed ? (
          <div className="flex justify-center py-1">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                modelStatus?.available ? 'bg-teal-400' : 'bg-slate-500'
              }`}
              title={modelStatus?.available ? 'Engine Online' : 'Engine Standby'}
            />
          </div>
        ) : (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] text-slate-300">
              <span className="flex items-center gap-1.5 text-slate-400 font-medium">
                <Radio className="w-3 h-3 text-teal-400" />
                <span>Engine</span>
              </span>
              <span className="inline-flex items-center gap-1 text-slate-400 font-medium text-[10px]">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    modelStatus?.available ? 'bg-teal-400 animate-pulse' : 'bg-slate-500'
                  }`}
                />
                <span>{modelStatus?.available ? (modelStatus?.mode === 'demo' ? 'Demo' : 'Online') : 'Standby'}</span>
              </span>
            </div>

            <div className="px-2 py-1.5 rounded-[6px] bg-white/5 border border-white/5 text-[10px] text-slate-400 flex items-center justify-between">
              <span className="truncate">{modelStatus?.model_name || 'ResNet-18'}</span>
              <span className="text-[9px] font-mono text-teal-400/90">{modelStatus?.model_version || 'v1'}</span>
            </div>

            <div className="flex items-center gap-1 text-[9px] text-slate-400 pt-0.5">
              <ShieldCheck className="w-3 h-3 text-teal-400 flex-shrink-0" />
              <span className="truncate">PneumoniaMNIST+ (224&times;224)</span>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
