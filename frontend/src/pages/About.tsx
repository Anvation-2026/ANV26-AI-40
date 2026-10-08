import React from 'react';
import {
  ShieldAlert,
  Database,
  Cpu,
  CheckCircle2,
  XCircle,
} from 'lucide-react';

export const About: React.FC = () => {
  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-black text-slate-900 tracking-tight">
          Methodology & System Architecture
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Detailed architectural principles, dataset provenance, uncertainty calibration, and safety limitations of the MedGuard AI educational prototype.
        </p>
      </div>

      {/* Mandatory Scope Card */}
      <div className="bg-amber-50 border border-amber-200 rounded-2xl p-6 text-amber-900">
        <div className="flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
          <div className="text-xs sm:text-sm leading-relaxed">
            <h2 className="font-bold text-amber-950 mb-1">
              Educational Research Prototype Scope
            </h2>
            <p>
              MedGuard AI is designed strictly for classroom instruction, medical informatics research, and artificial intelligence safety demonstration. It is <strong>not a medical device</strong>, has no FDA, CE, or regulatory clearance, and must never be used for diagnostic, prognostic, or treatment decisions.
            </p>
          </div>
        </div>
      </div>

      {/* Supported Modality & Dataset Characteristics */}
      <section className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-4">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <Database className="w-5 h-5 text-teal-600" />
          <span>Dataset & Domain Specification</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/80 space-y-1.5">
            <span className="font-bold text-slate-800 block text-sm">
              PneumoniaMNIST+ Benchmark
            </span>
            <p className="text-slate-600">
              Derived from pediatric chest radiographs depicting normal lungs vs. viral/bacterial pneumonia.
            </p>
            <ul className="space-y-1 text-slate-500 pt-1">
              <li>&bull; Total Samples: <strong>5,856 radiographs</strong></li>
              <li>&bull; Image Resolution: <strong>224 &times; 224 pixels</strong></li>
              <li>&bull; Target Classes: <strong>Normal (0) vs Pneumonia (1)</strong></li>
            </ul>
          </div>

          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/80 space-y-1.5">
            <span className="font-bold text-slate-800 block text-sm">
              Standardized Partitions
            </span>
            <p className="text-slate-600">
              Rigidly split into train, validation, and test subsets to evaluate generalization:
            </p>
            <ul className="space-y-1 text-slate-500 pt-1">
              <li>&bull; Training Set: <strong>4,708 samples (80.4%)</strong></li>
              <li>&bull; Validation Set: <strong>524 samples (8.9%)</strong></li>
              <li>&bull; Test Benchmark: <strong>624 samples (10.7%)</strong></li>
            </ul>
          </div>
        </div>
      </section>

      {/* Decision Support & Safety Principles */}
      <section className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-5">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-teal-600" />
          <span>Interpretability & Triage Principles</span>
        </h2>

        <div className="space-y-4 text-xs sm:text-sm text-slate-600 leading-relaxed">
          <div>
            <h3 className="font-bold text-slate-800 text-sm mb-1">
              1. Calibrated Confidence vs Raw Softmax
            </h3>
            <p>
              Standard neural network softmax logits often suffer from overconfidence on ambiguous samples. MedGuard AI integrates temperature scaling and Platt scaling so that a 90% confidence output reflects approximately 90% empirical accuracy on held-out benchmarks.
            </p>
          </div>

          <div>
            <h3 className="font-bold text-slate-800 text-sm mb-1">
              2. Safety Abstention (Selective Classification)
            </h3>
            <p>
              Rather than forcing a prediction on uncertain inputs, the system computes Monte Carlo Dropout or entropy metrics. If uncertainty surpasses safety thresholds, the engine abstains from committing to a finding, suppresses probability figures, and routes directly to human expert review.
            </p>
          </div>

          <div>
            <h3 className="font-bold text-slate-800 text-sm mb-1">
              3. Grad-CAM Saliency Maps & Limitations
            </h3>
            <p>
              Gradient-weighted Class Activation Mapping (Grad-CAM) visualizes coarse pixel regions that produced high gradient activations in the final convolutional layer. Grad-CAM indicates <em>where the network looked</em>, but does not provide anatomical verification or boundary delineation of consolidations.
            </p>
          </div>

          <div>
            <h3 className="font-bold text-slate-800 text-sm mb-1">
              4. Quality & Out-of-Distribution (OOD) Guardrails
            </h3>
            <p>
              Laplacian blur variance and histogram contrast estimators check upload clarity. A Mahalanobis distance OOD filter detects when an uploaded image (such as a CT scan, bone radiograph, or non-medical photo) falls outside chest X-ray distribution.
            </p>
          </div>
        </div>
      </section>

      {/* In-Scope vs Out-of-Scope Checklist */}
      <section className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-4">
        <h2 className="text-base font-bold text-slate-900">
          Scope Boundaries
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div className="p-4 bg-emerald-50/60 rounded-xl border border-emerald-200/80 space-y-2">
            <span className="font-bold text-emerald-950 flex items-center gap-1.5 text-sm">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>Intended In-Scope Use</span>
            </span>
            <ul className="space-y-1.5 text-emerald-900">
              <li>&bull; Classroom demonstrations of medical AI safety concepts</li>
              <li>&bull; Comparing raw vs calibrated model probabilities</li>
              <li>&bull; Illustrating model abstention on ambiguous radiographs</li>
              <li>&bull; Evaluating Grad-CAM visual attention overlays</li>
              <li>&bull; Testing heuristic image quality degradation</li>
            </ul>
          </div>

          <div className="p-4 bg-rose-50/60 rounded-xl border border-rose-200/80 space-y-2">
            <span className="font-bold text-rose-950 flex items-center gap-1.5 text-sm">
              <XCircle className="w-4 h-4 text-rose-600" />
              <span>Strictly Out-of-Scope Use</span>
            </span>
            <ul className="space-y-1.5 text-rose-900">
              <li>&bull; Diagnostic decisions for patients in any clinical setting</li>
              <li>&bull; Emergency room triage prioritization or scoring</li>
              <li>&bull; Non-chest imaging modalities (CT, MRI, ultrasound)</li>
              <li>&bull; Detecting pathologies outside binary normal vs pneumonia</li>
              <li>&bull; Uploading protected health information (PHI)</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
};
