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
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-navy-foreground tracking-tight">
          Methodology, Calibration & Trust Boundaries
        </h1>
        <p className="text-xs text-navy-muted mt-0.5">
          Technical specifications, dataset parameters, uncertainty estimation, and safety boundary conditions.
        </p>
      </div>

      {/* Mandatory Scope Card */}
      <div className="bg-[#FFFBEB] border border-amber-200 rounded-[12px] p-5 text-[#92400E]">
        <div className="flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-clinical-warning flex-shrink-0 mt-0.5" />
          <div className="text-xs leading-relaxed">
            <h2 className="font-bold text-[#78350F] mb-0.5">
              Educational & Scientific Research System Scope
            </h2>
            <p>
              MedGuard AI is developed strictly for healthcare informatics instruction and AI safety methodology demonstrations. It is <strong>not a medical device</strong>, has no FDA, CE, or regulatory clearance, and must never be used for diagnostic, prognostic, or clinical treatment decisions.
            </p>
          </div>
        </div>
      </div>

      {/* Supported Modality & Dataset Specifications */}
      <section className="bg-surface rounded-[12px] border border-border p-5 shadow-xs space-y-4">
        <h2 className="text-sm font-bold text-navy-foreground flex items-center gap-2">
          <Database className="w-4 h-4 text-teal-700" />
          <span>Dataset & Imaging Modality Parameters</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 text-xs">
          <div className="p-4 bg-canvas rounded-[10px] border border-border space-y-1.5">
            <span className="font-bold text-navy-foreground block text-xs">
              PneumoniaMNIST+ Benchmark Dataset
            </span>
            <p className="text-navy-muted">
              Originates from pediatric anterior-posterior chest radiographs labeled for normal lungs versus pneumonia.
            </p>
            <ul className="space-y-1 text-navy-muted pt-1 text-[11px]">
              <li>&bull; Total Partition Samples: <strong className="text-navy-foreground">5,856 radiographs</strong></li>
              <li>&bull; Standard Input Matrix: <strong className="text-navy-foreground">224 &times; 224 pixels</strong></li>
              <li>&bull; Target Classes: <strong className="text-navy-foreground">Normal (0) &middot; Pneumonia (1)</strong></li>
            </ul>
          </div>

          <div className="p-4 bg-canvas rounded-[10px] border border-border space-y-1.5">
            <span className="font-bold text-navy-foreground block text-xs">
              Rigid Benchmark Partitioning
            </span>
            <p className="text-navy-muted">
              Pre-split to evaluate out-of-sample generalization without label leakage:
            </p>
            <ul className="space-y-1 text-navy-muted pt-1 text-[11px]">
              <li>&bull; Training Set: <strong className="text-navy-foreground">4,708 samples (80.4%)</strong></li>
              <li>&bull; Validation Set: <strong className="text-navy-foreground">524 samples (8.9%)</strong></li>
              <li>&bull; Independent Test Set: <strong className="text-navy-foreground">624 samples (10.7%)</strong></li>
            </ul>
          </div>
        </div>
      </section>

      {/* Core Safety & Explainability Principles */}
      <section className="bg-surface rounded-[12px] border border-border p-5 shadow-xs space-y-4">
        <h2 className="text-sm font-bold text-navy-foreground flex items-center gap-2">
          <Cpu className="w-4 h-4 text-teal-700" />
          <span>Decision Support Architecture & Safeguards</span>
        </h2>

        <div className="space-y-3.5 text-xs text-navy-muted leading-relaxed">
          <div className="p-3.5 bg-canvas rounded-[8px] border border-border/70">
            <h3 className="font-bold text-navy-foreground text-xs mb-1">
              1. Calibrated Probabilities vs. Raw Softmax Scores
            </h3>
            <p>
              Standard neural network softmax probabilities tend to be overly confident on ambiguous inputs. The engine uses Platt scaling and temperature calibration so that a reported 90% confidence matches approximately 90% empirical precision on test cohorts.
            </p>
          </div>

          <div className="p-3.5 bg-canvas rounded-[8px] border border-border/70">
            <h3 className="font-bold text-navy-foreground text-xs mb-1">
              2. Selective Classification & Uncertainty Abstention
            </h3>
            <p>
              Rather than guessing when an image is ambiguous, the system computes Monte Carlo Dropout ensemble variance. When uncertainty exceeds safe limits, it triggers an educational abstention, suppresses probability figures, and routes directly to human expert review.
            </p>
          </div>

          <div className="p-3.5 bg-canvas rounded-[8px] border border-border/70">
            <h3 className="font-bold text-navy-foreground text-xs mb-1">
              3. Grad-CAM Activation Visualizations
            </h3>
            <p>
              Gradient-weighted Class Activation Mapping computes coarse 2D heatmaps showing convolutional layers with high activation. This reveals where the model looked, but does not provide clinically verified anatomical segmentations.
            </p>
          </div>

          <div className="p-3.5 bg-canvas rounded-[8px] border border-border/70">
            <h3 className="font-bold text-navy-foreground text-xs mb-1">
              4. Pre-Inference Quality & Domain Filters
            </h3>
            <p>
              Laplacian blur variance and histogram contrast estimators check upload clarity. A Mahalanobis distance filter halts inference if a non-chest image is submitted.
            </p>
          </div>
        </div>
      </section>

      {/* In-Scope vs Out-of-Scope Checklist */}
      <section className="bg-surface rounded-[12px] border border-border p-5 shadow-xs space-y-3">
        <h2 className="text-sm font-bold text-navy-foreground">
          Clinical Boundary Conditions
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 text-xs">
          <div className="p-4 bg-emerald-50/50 rounded-[10px] border border-emerald-200 space-y-2">
            <span className="font-bold text-clinical-success flex items-center gap-1.5 text-xs">
              <CheckCircle2 className="w-4 h-4" />
              <span>Intended In-Scope Use</span>
            </span>
            <ul className="space-y-1 text-navy-foreground text-[11px] leading-relaxed">
              <li>&bull; Medical informatics classroom demonstrations</li>
              <li>&bull; Demonstrating probability calibration vs uncalibrated logits</li>
              <li>&bull; Evaluating model abstention on degraded radiographs</li>
              <li>&bull; Inspecting Grad-CAM activation attention patterns</li>
            </ul>
          </div>

          <div className="p-4 bg-rose-50/50 rounded-[10px] border border-rose-200 space-y-2">
            <span className="font-bold text-clinical-danger flex items-center gap-1.5 text-xs">
              <XCircle className="w-4 h-4" />
              <span>Strictly Out-of-Scope Use</span>
            </span>
            <ul className="space-y-1 text-navy-foreground text-[11px] leading-relaxed">
              <li>&bull; Diagnostic decisions for patients in any clinical setting</li>
              <li>&bull; Emergency room triage scoring or prioritization</li>
              <li>&bull; Non-chest imaging (CT scans, MRIs, ultrasounds)</li>
              <li>&bull; Uploading protected health information (PHI)</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
};
