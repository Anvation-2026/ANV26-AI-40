import React from 'react';
import { Link } from 'react-router-dom';
import {
  ShieldAlert,
  ArrowRight,
  Eye,
  Activity,
  HelpCircle,
  FileCheck,
} from 'lucide-react';
import { ModelStatusCard } from '../components/ModelStatusCard';

export const Dashboard: React.FC = () => {
  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-teal-900 via-teal-800 to-slate-900 text-white rounded-3xl p-8 sm:p-12 shadow-md relative overflow-hidden">
        <div className="relative z-10 max-w-2xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-700/60 border border-teal-500/30 text-teal-200 text-xs font-semibold mb-4">
            <Activity className="w-3.5 h-3.5" />
            <span>Educational Research Prototype &middot; PneumoniaMNIST+</span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-black tracking-tight text-white leading-tight">
            Evidence-Based Chest X-Ray Triage Support
          </h1>
          <p className="mt-4 text-base text-teal-100 leading-relaxed">
            MedGuard AI provides transparent educational decision support for pediatric and adult chest X-rays. It combines predictive modeling with safety abstention, image quality verification, out-of-distribution detection, and Grad-CAM visual evidence.
          </p>

          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              to="/analyze"
              className="inline-flex items-center gap-2 px-6 py-3 bg-teal-500 hover:bg-teal-400 text-slate-950 text-sm font-bold rounded-xl shadow transition-all focus:outline-none focus:ring-4 focus:ring-teal-300"
            >
              <span>Analyze an Image</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/validation"
              className="inline-flex items-center gap-2 px-5 py-3 bg-white/10 hover:bg-white/20 text-white text-sm font-semibold rounded-xl border border-white/20 transition-all focus:outline-none focus:ring-4 focus:ring-white/20"
            >
              <span>View Model Validation</span>
            </Link>
          </div>
        </div>

        {/* Decorative background element */}
        <div className="absolute -right-20 -bottom-20 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl pointer-events-none"></div>
      </section>

      {/* Mandatory Scope & Safety Notice */}
      <section className="bg-amber-50 border border-amber-200 rounded-2xl p-5 flex items-start gap-3.5 text-amber-900 shadow-2xs">
        <ShieldAlert className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
        <div className="text-xs sm:text-sm leading-relaxed">
          <span className="font-bold block text-amber-950 mb-0.5">
            Educational & Instructional Scope Notice
          </span>
          MedGuard AI is an experimental educational decision-support prototype. It is{' '}
          <strong>not certified as a medical device</strong>, has not received regulatory clearance, and must not be used for diagnosis, clinical staging, or patient treatment decisions. Uploads must consist strictly of public or de-identified educational images.
        </div>
      </section>

      {/* Model Engine Status Card */}
      <section>
        <ModelStatusCard />
      </section>

      {/* Core Safety Architectural Pillars */}
      <section className="space-y-4">
        <h2 className="text-lg font-bold text-slate-900 tracking-tight">
          Safety & Explainability Architecture
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
            <div className="w-10 h-10 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center mb-3">
              <FileCheck className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-slate-900 mb-1">
              Quality & OOD Filtering
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Heuristic checks detect motion blur, improper exposure, and anatomical out-of-distribution inputs prior to model inference.
            </p>
          </div>

          <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
            <div className="w-10 h-10 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center mb-3">
              <HelpCircle className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-slate-900 mb-1">
              Safety Abstention Policy
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              When uncertainty exceeds safe thresholds, the system deliberately abstains from outputting a definitive finding and mandates human review.
            </p>
          </div>

          <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
            <div className="w-10 h-10 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center mb-3">
              <Eye className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-slate-900 mb-1">
              Grad-CAM Visual Overlays
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Gradient-weighted Class Activation Maps highlight regions influencing classification, providing explainable inspection for students.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
