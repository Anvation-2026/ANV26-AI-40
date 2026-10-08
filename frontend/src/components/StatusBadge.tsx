import React from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  ImageOff,
  Compass,
  AlertOctagon,
  XCircle,
  HelpCircle,
} from 'lucide-react';
import { AnalysisStatus } from '../types/analysis';

interface StatusBadgeProps {
  status: AnalysisStatus;
  size?: 'sm' | 'md' | 'lg';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'md' }) => {
  const config = {
    success: {
      label: 'Analysis Successful',
      bg: 'bg-emerald-50 text-emerald-800 border-emerald-300',
      icon: CheckCircle2,
    },
    uncertain: {
      label: 'Abstained / Uncertain',
      bg: 'bg-amber-50 text-amber-800 border-amber-300',
      icon: AlertTriangle,
    },
    poor_quality: {
      label: 'Poor Image Quality',
      bg: 'bg-orange-50 text-orange-800 border-orange-300',
      icon: ImageOff,
    },
    ood: {
      label: 'Out of Distribution',
      bg: 'bg-purple-50 text-purple-800 border-purple-300',
      icon: Compass,
    },
    model_unavailable: {
      label: 'Model Unavailable',
      bg: 'bg-rose-50 text-rose-800 border-rose-300',
      icon: AlertOctagon,
    },
    invalid_input: {
      label: 'Invalid Input',
      bg: 'bg-rose-50 text-rose-800 border-rose-300',
      icon: XCircle,
    },
    error: {
      label: 'System Error',
      bg: 'bg-red-50 text-red-800 border-red-300',
      icon: AlertOctagon,
    },
  }[status] || {
    label: status,
    bg: 'bg-slate-100 text-slate-800 border-slate-300',
    icon: HelpCircle,
  };

  const Icon = config.icon;

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5 gap-1',
    md: 'text-sm px-2.5 py-1 gap-1.5',
    lg: 'text-base px-3.5 py-1.5 gap-2 font-semibold',
  }[size];

  const iconSizes = {
    sm: 'w-3.5 h-3.5',
    md: 'w-4 h-4',
    lg: 'w-5 h-5',
  }[size];

  return (
    <span
      className={`inline-flex items-center font-medium rounded-full border ${config.bg} ${sizeClasses} shadow-2xs`}
    >
      <Icon className={`${iconSizes} flex-shrink-0`} aria-hidden="true" />
      <span>{config.label}</span>
    </span>
  );
};
