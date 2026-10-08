export type AnalysisStatus =
  | 'success'
  | 'uncertain'
  | 'poor_quality'
  | 'ood'
  | 'model_unavailable'
  | 'invalid_input'
  | 'error';

export interface QualityCheck {
  value: number | null;
  threshold: number | null;
  passed: boolean | null;
}

export interface QualityInfo {
  evaluated: boolean;
  status: 'acceptable' | 'poor' | 'not_evaluated';
  blur: QualityCheck | null;
  brightness: QualityCheck | null;
  contrast: QualityCheck | null;
  reasons: string[];
}

export interface OODInfo {
  evaluated: boolean;
  is_ood: boolean | null;
  score: number | null;
  method: string | null;
  reason: string | null;
}

export interface UncertaintyInfo {
  level: 'low' | 'moderate' | 'high' | 'not_evaluated';
  value: number | null;
  method: string | null;
}

export interface HeatmapInfo {
  available: boolean;
  data_url: string | null;
  kind: 'overlay' | 'heatmap_only' | null;
  message: string | null;
}

export interface ModelMeta {
  available: boolean;
  model_name: string | null;
  model_version: string | null;
  dataset: string | null;
  supported_modality?: string | null;
  calibration_available?: boolean | null;
  ood_available?: boolean | null;
  quality_available?: boolean | null;
  gradcam_available?: boolean | null;
}

export interface TriageInfo {
  action:
    | 'expert_review_recommended'
    | 'expert_review_required_abstained'
    | 'resubmit_better_sample'
    | 'unsupported_input'
    | 'technical_error';
  title: string;
  message: string;
  reasons: string[];
}

export interface ErrorInfo {
  code: string;
  message: string;
}

export interface AnalysisResponse {
  request_id: string;
  status: AnalysisStatus;
  mode: 'real' | 'demo';
  finding: 'normal' | 'pneumonia' | null;
  abstained: boolean;
  raw_score: number | null;
  probability: number | null; // calibrated
  probability_of: string | null;
  uncertainty: UncertaintyInfo;
  quality: QualityInfo;
  ood: OODInfo;
  heatmap: HeatmapInfo;
  triage: TriageInfo;
  explanation: string;
  evidence: string[];
  limitations: string[];
  model: ModelMeta;
  error: ErrorInfo | null;
  disclaimer: string;
}

export interface HealthResponse {
  status: 'ok';
  mode: 'real' | 'demo';
  version: string;
  ml_module_loaded: boolean;
  timestamp: string;
}

export interface ModelStatusResponse {
  available: boolean;
  model_name: string | null;
  model_version: string | null;
  supported_image_type: string;
  inference_ready: boolean;
  calibration_available: boolean;
  ood_available: boolean;
  quality_available: boolean;
  gradcam_available: boolean;
  mode: 'real' | 'demo';
  message: string | null;
}

export interface ValidationMetrics {
  accuracy: number | null;
  sensitivity: number | null;
  specificity: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  auroc: number | null;
  false_negative_rate: number | null;
  ece: number | null;
  abstention_coverage: number | null;
  rejection_rate: number | null;
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
}

export interface ValidationReport {
  model_name: string | null;
  model_version: string | null;
  dataset: string | null;
  evaluated_on: string | null;
  split_counts: Record<string, number> | null;
  metrics: ValidationMetrics | null;
  confusion_matrix: ConfusionMatrixData | null;
  limitations: string[];
  generated_at: string | null;
}

export interface ValidationResponse {
  status: 'available' | 'pending' | 'error';
  report: ValidationReport | null;
  message: string | null;
}
