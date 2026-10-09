import {
  AnalysisResponse,
  HealthResponse,
  ModelStatusResponse,
  ValidationResponse,
  FractureStatusResponse,
} from '../types/analysis';

const ENV_API_BASE = import.meta.env.VITE_API_BASE || '';

export function getApiBase(): string {
  if (typeof window !== 'undefined') {
    const custom = localStorage.getItem('medguard_api_base');
    if (custom) return custom.replace(/\/$/, '');
  }
  return ENV_API_BASE;
}

export function setApiBase(url: string): void {
  if (typeof window !== 'undefined') {
    if (!url) {
      localStorage.removeItem('medguard_api_base');
    } else {
      localStorage.setItem('medguard_api_base', url.replace(/\/$/, ''));
    }
  }
}

export function isStaticHostWithoutBackend(): boolean {
  if (typeof window === 'undefined') return false;
  const currentApi = getApiBase();
  if (currentApi) return false; // If custom/env backend URL is set, attempt live backend
  // On GitHub Pages or static host without local backend, avoid sending doomed requests to static file server
  return window.location.hostname.endsWith('github.io');
}

export class NetworkError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'NetworkError';
  }
}

// -------------------------------------------------------------
// Interactive Client-Side Demo Fixtures (for GitHub Pages / Vercel static)
// -------------------------------------------------------------

const DEMO_LIMITATIONS = [
  'Interactive client-side educational prototype for portfolio and architecture verification.',
  'Not certified as a medical diagnostic device for patient clinical decision-making.',
  'Trained on educational research benchmarks (PneumoniaMNIST+ / MURA subsets).',
  'All probabilities and entropy metrics are calibrated against educational research distributions.',
];

function generateDemoPrediction(filename: string, scenario?: string): AnalysisResponse {
  const name = filename.toLowerCase();
  const selected = (scenario || '').toLowerCase();

  let isPneumonia = false;
  let isUncertain = false;
  let isPoorQuality = false;
  let isOOD = false;

  if (selected === 'poor_quality' || name.includes('blur') || name.includes('poor')) {
    isPoorQuality = true;
  } else if (selected === 'ood' || name.includes('ood') || name.includes('cat') || name.includes('dog')) {
    isOOD = true;
  } else if (selected === 'uncertain' || name.includes('uncertain')) {
    isUncertain = true;
  } else if (selected === 'pneumonia' || name.includes('pneumonia')) {
    isPneumonia = true;
  } else if (selected === 'normal' || name.includes('normal')) {
    isPneumonia = false;
  } else {
    // Default fallback based on filename length hash for consistency
    isPneumonia = name.length % 2 === 0;
  }

  const reqId = 'demo-' + Math.random().toString(36).substring(2, 9);

  if (isPoorQuality) {
    return {
      request_id: reqId,
      status: 'poor_quality',
      mode: 'demo',
      finding: null,
      abstained: true,
      raw_score: null,
      probability: null,
      probability_of: null,
      uncertainty: { level: 'not_evaluated', value: null, method: null },
      quality: {
        evaluated: true,
        status: 'poor',
        blur: { value: 38.4, threshold: 100.0, passed: false },
        brightness: { value: 24.1, threshold: 40.0, passed: false },
        contrast: { value: 16.0, threshold: 20.0, passed: false },
        reasons: ['Image blur score (38.4) below acceptable diagnostic threshold (100.0)'],
      },
      ood: { evaluated: false, is_ood: null, score: null, method: null, reason: null },
      heatmap: { available: false, data_url: null, kind: null, message: 'Heatmap skipped for degraded quality image' },
      triage: {
        action: 'resubmit_better_sample',
        title: 'Image Quality Safeguard Triggered (Demo)',
        message: 'Educational safeguard: input image resolution or exposure failed quality filters. Resubmit a clear radiograph.',
        reasons: ['Significant blur or artifacting detected by automated pre-screening.'],
      },
      explanation: 'Pre-analysis quality assessment detected severe motion blur or low contrast, preventing safe inference.',
      evidence: [
        'Quality gate: Laplacian blur index failed threshold',
        'Model inference bypassed to avoid erroneous clinical extrapolation',
      ],
      limitations: DEMO_LIMITATIONS,
      model: {
        available: true,
        model_name: 'MedGuard-ResNet18 (Interactive Demo)',
        model_version: 'v1.0-demo',
        dataset: 'PneumoniaMNIST+ (224x224)',
        supported_modality: 'Chest X-ray (educational)',
      },
      error: null,
      disclaimer: 'Educational prototype only — interactive demo fixture.',
    };
  }

  if (isOOD) {
    return {
      request_id: reqId,
      status: 'ood',
      mode: 'demo',
      finding: null,
      abstained: true,
      raw_score: null,
      probability: null,
      probability_of: null,
      uncertainty: { level: 'not_evaluated', value: null, method: null },
      quality: {
        evaluated: true,
        status: 'acceptable',
        blur: { value: 210.0, threshold: 100.0, passed: true },
        brightness: { value: 125.0, threshold: 40.0, passed: true },
        contrast: { value: 68.0, threshold: 20.0, passed: true },
        reasons: [],
      },
      ood: {
        evaluated: true,
        is_ood: true,
        score: 54.8,
        method: 'Mahalanobis Distance',
        reason: 'Feature space distance (54.8) exceeded pediatric chest radiograph distribution envelope (35.0)',
      },
      heatmap: { available: false, data_url: null, kind: null, message: 'Heatmap withheld for out-of-distribution input' },
      triage: {
        action: 'unsupported_input',
        title: 'Out-of-Distribution Modality (Demo)',
        message: 'Educational safeguard: this image does not match the distribution of pediatric frontal chest radiographs.',
        reasons: ['Embedding vector lies outside known clinical latent clusters.'],
      },
      explanation: 'Deep latent distribution check flagged high anomaly score; automated evaluation safely abstained.',
      evidence: [
        'Mahalanobis distance: 54.8 (critical limit: 35.0)',
        'Sample rejected to prevent silent hallucination on non-chest or altered imagery',
      ],
      limitations: DEMO_LIMITATIONS,
      model: {
        available: true,
        model_name: 'MedGuard-ResNet18 (Interactive Demo)',
        model_version: 'v1.0-demo',
        dataset: 'PneumoniaMNIST+ (224x224)',
        supported_modality: 'Chest X-ray (educational)',
      },
      error: null,
      disclaimer: 'Educational prototype only — interactive demo fixture.',
    };
  }

  if (isUncertain) {
    return {
      request_id: reqId,
      status: 'uncertain',
      mode: 'demo',
      finding: null,
      abstained: true,
      raw_score: 0.448,
      probability: null,
      probability_of: null,
      uncertainty: { level: 'high', value: 0.9969, method: 'Normalized Entropy' },
      quality: {
        evaluated: true,
        status: 'acceptable',
        blur: { value: 189.5, threshold: 100.0, passed: true },
        brightness: { value: 130.2, threshold: 40.0, passed: true },
        contrast: { value: 62.4, threshold: 20.0, passed: true },
        reasons: [],
      },
      ood: { evaluated: true, is_ood: false, score: 18.2, method: 'Mahalanobis', reason: null },
      heatmap: {
        available: true,
        data_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
        kind: 'overlay',
        message: 'Educational ambiguity attention map generated for borderline regions.',
      },
      triage: {
        action: 'expert_review_required_abstained',
        title: 'High Uncertainty — Decision Abstained (Demo)',
        message: 'Model confidence fell below the safety operating threshold (tau = 0.56). Expert radiological review required.',
        reasons: ['Calibrated prediction falls in borderline ambiguity interval (0.45 – 0.55).'],
      },
      explanation: 'The neural network exhibited elevated epistemic entropy, indicating ambiguous clinical features.',
      evidence: [
        'Calibrated prediction: 0.49 (borderline overlap zone)',
        'Normalized predictive entropy: 0.9969 (threshold limit: 0.75)',
        'Ambiguity attention map highlights conflicting feature contributions',
      ],
      limitations: DEMO_LIMITATIONS,
      model: {
        available: true,
        model_name: 'MedGuard-ResNet18 (Interactive Demo)',
        model_version: 'v1.0-demo',
        dataset: 'PneumoniaMNIST+ (224x224)',
        supported_modality: 'Chest X-ray (educational)',
      },
      error: null,
      disclaimer: 'Educational prototype only — interactive demo fixture.',
    };
  }

  // Definite Pneumonia
  if (isPneumonia) {
    return {
      request_id: reqId,
      status: 'success',
      mode: 'demo',
      finding: 'pneumonia',
      abstained: false,
      raw_score: 0.884,
      probability: 0.87,
      probability_of: 'pneumonia',
      uncertainty: { level: 'low', value: 0.042, method: 'Normalized Entropy' },
      quality: {
        evaluated: true,
        status: 'acceptable',
        blur: { value: 240.5, threshold: 100.0, passed: true },
        brightness: { value: 128.0, threshold: 40.0, passed: true },
        contrast: { value: 65.2, threshold: 20.0, passed: true },
        reasons: [],
      },
      ood: { evaluated: true, is_ood: false, score: 14.1, method: 'Mahalanobis', reason: null },
      heatmap: {
        available: true,
        data_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
        kind: 'overlay',
        message: 'Grad-CAM overlay highlighting regions with high focal feature weighting.',
      },
      triage: {
        action: 'expert_review_recommended',
        title: 'Pneumonia Infiltrate Pattern Detected (Demo)',
        message: 'Educational finding: focal parenchymal consolidation identified in lung fields with high confidence.',
        reasons: ['Calibrated probability (87.0%) exceeds priority triage threshold.'],
      },
      explanation: 'Pronounced focal density and alveolar opacities observed consistent with lobar parenchymal infiltrate.',
      evidence: [
        'Calibrated P(Pneumonia): 0.870 (high certainty)',
        'Image quality: acceptable (blur score 240.5)',
        'In-distribution confirmation: Mahalanobis score 14.1',
        'Grad-CAM attention concentrated in lower lobe airspace region',
      ],
      limitations: DEMO_LIMITATIONS,
      model: {
        available: true,
        model_name: 'MedGuard-ResNet18 (Interactive Demo)',
        model_version: 'v1.0-demo',
        dataset: 'PneumoniaMNIST+ (224x224)',
        supported_modality: 'Chest X-ray (educational)',
      },
      error: null,
      disclaimer: 'Educational prototype only — interactive demo fixture.',
    };
  }

  // Definite Normal
  return {
    request_id: reqId,
    status: 'success',
    mode: 'demo',
    finding: 'normal',
    abstained: false,
    raw_score: 0.115,
    probability: 0.912,
    probability_of: 'normal',
    uncertainty: { level: 'low', value: 0.031, method: 'Normalized Entropy' },
    quality: {
      evaluated: true,
      status: 'acceptable',
      blur: { value: 265.0, threshold: 100.0, passed: true },
      brightness: { value: 135.0, threshold: 40.0, passed: true },
      contrast: { value: 70.1, threshold: 20.0, passed: true },
      reasons: [],
    },
    ood: { evaluated: true, is_ood: false, score: 9.8, method: 'Mahalanobis', reason: null },
    heatmap: {
      available: true,
      data_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
      kind: 'overlay',
      message: 'Grad-CAM visual overlay reflecting symmetric bilateral aeration.',
    },
    triage: {
      action: 'expert_review_recommended',
      title: 'Clear Aerated Lung Parenchyma (Demo)',
      message: 'Educational finding: bilateral lung fields demonstrate clear, uniform aeration without focal consolidation.',
      reasons: ['Calibrated normal probability (91.2%) with low entropy.'],
    },
    explanation: 'Symmetric bilateral radiolucency observed with clear costophrenic angles and no consolidation.',
    evidence: [
      'Calibrated P(Normal): 0.912',
      'Image quality check: passed (blur 265.0, contrast 70.1)',
      'In-distribution check: passed (score 9.8)',
    ],
    limitations: DEMO_LIMITATIONS,
    model: {
      available: true,
      model_name: 'MedGuard-ResNet18 (Interactive Demo)',
      model_version: 'v1.0-demo',
      dataset: 'PneumoniaMNIST+ (224x224)',
      supported_modality: 'Chest X-ray (educational)',
    },
    error: null,
    disclaimer: 'Educational prototype only — interactive demo fixture.',
  };
}

// -------------------------------------------------------------
// API Service Methods with Fallback
// -------------------------------------------------------------

async function handleResponse<T>(res: Response, endpointDesc: string): Promise<T> {
  const contentType = res.headers.get('content-type') || '';
  if (!contentType.includes('application/json')) {
    throw new NetworkError('API endpoint returned non-JSON response.');
  }

  const data = await res.json();
  if (!res.ok) {
    const errorMsg = data?.message || data?.detail || `${endpointDesc} returned status ${res.status}`;
    throw new NetworkError(errorMsg);
  }
  return data as T;
}

export async function getHealth(): Promise<HealthResponse> {
  if (isStaticHostWithoutBackend()) {
    return {
      status: 'ok',
      mode: 'demo',
      version: '1.0.0-demo',
      ml_module_loaded: true,
      timestamp: new Date().toISOString(),
    };
  }

  const apiBase = getApiBase();
  try {
    const res = await fetch(`${apiBase}/api/health`);
    if (res.ok) {
      return await handleResponse<HealthResponse>(res, 'Health check');
    }
  } catch {
    // Network offline / fallback to demo
  }

  // Graceful Demo Mode fallback
  return {
    status: 'ok',
    mode: 'demo',
    version: '1.0.0-demo',
    ml_module_loaded: true,
    timestamp: new Date().toISOString(),
  };
}

export async function getModelStatus(): Promise<ModelStatusResponse> {
  if (isStaticHostWithoutBackend()) {
    return {
      available: true,
      model_name: 'ResNet-18 (Interactive Demo)',
      model_version: 'v1.0-demo',
      supported_image_type: 'Chest X-ray (educational)',
      inference_ready: true,
      calibration_available: true,
      ood_available: true,
      quality_available: true,
      gradcam_available: true,
      mode: 'demo',
      message: 'Client-side interactive demo active (ready for evaluation)',
    };
  }

  const apiBase = getApiBase();
  try {
    const res = await fetch(`${apiBase}/api/model/status`);
    if (res.ok) {
      return await handleResponse<ModelStatusResponse>(res, 'Model status');
    }
  } catch {
    // Fall through to demo status
  }

  return {
    available: true,
    model_name: 'ResNet-18 (Interactive Demo)',
    model_version: 'v1.0-demo',
    supported_image_type: 'Chest X-ray (educational)',
    inference_ready: true,
    calibration_available: true,
    ood_available: true,
    quality_available: true,
    gradcam_available: true,
    mode: 'demo',
    message: 'Client-side interactive demo active (ready for evaluation)',
  };
}

export async function getValidation(): Promise<ValidationResponse> {
  if (!isStaticHostWithoutBackend()) {
    const apiBase = getApiBase();
    try {
      const res = await fetch(`${apiBase}/api/validation`);
      if (res.ok) {
        return await handleResponse<ValidationResponse>(res, 'Validation report');
      }
    } catch {
      // Fall through to embedded report
    }
  }

  return {
    status: 'available',
    report: {
      model_name: 'ResNet-18 (PneumoniaMNIST+)',
      model_version: 'v1+2c4f0760',
      dataset: 'PneumoniaMNIST+ (224x224)',
      evaluated_on: 'Independent Test Split (n=624)',
      split_counts: { train: 4708, val: 524, test: 624 },
      metrics: {
        accuracy: 0.9087,
        sensitivity: 0.9974,
        specificity: 0.7607,
        precision: 0.8742,
        recall: 0.9974,
        f1: 0.9317,
        auroc: 0.9795,
        false_negative_rate: 0.0026,
        ece: 0.0384,
        abstention_coverage: 0.942,
        rejection_rate: 0.058,
      },
      confusion_matrix: {
        labels: ['Normal', 'Pneumonia'],
        matrix: [
          [178, 56],
          [1, 389],
        ],
      },
      limitations: DEMO_LIMITATIONS,
      generated_at: '2026-10-08T12:00:44.740Z',
    },
    message: null,
  };
}

export async function predictImage(
  file: File,
  signal?: AbortSignal,
  demoScenario?: string
): Promise<AnalysisResponse> {
  if (!isStaticHostWithoutBackend()) {
    const formData = new FormData();
    formData.append('file', file);

    const headers: Record<string, string> = {};
    if (demoScenario) {
      headers['X-Demo-Scenario'] = demoScenario;
    }

    const apiBase = getApiBase();
    try {
      const res = await fetch(`${apiBase}/api/predict`, {
        method: 'POST',
        body: formData,
        headers,
        signal,
      });

      if (res.ok) {
        const data = await res.json();
        return data as AnalysisResponse;
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        throw err;
      }
      // Fall back to client-side demo inference
    }
  }

  // Simulate short network latency for realism
  await new Promise((resolve) => setTimeout(resolve, 350));
  return generateDemoPrediction(file.name, demoScenario);
}

export async function getFractureStatus(): Promise<FractureStatusResponse> {
  if (!isStaticHostWithoutBackend()) {
    const apiBase = getApiBase();
    try {
      const res = await fetch(`${apiBase}/api/fracture/status`);
      if (res.ok) {
        return await handleResponse<FractureStatusResponse>(res, 'Fracture model status');
      }
    } catch {
      // Fall through
    }
  }

  return {
    available: true,
    status: 'online',
    model_name: 'ConvNeXt-Base Bone Fracture (Demo)',
    backbone: 'convnext_base',
    device: 'cpu',
    supported_anatomies: ['hand', 'wrist', 'leg', 'foot', 'arm'],
    message: 'Fracture demo analysis module active',
  };
}

export async function predictFracture(
  file: File,
  signal?: AbortSignal
): Promise<any> {
  if (!isStaticHostWithoutBackend()) {
    const formData = new FormData();
    formData.append('file', file);

    const apiBase = getApiBase();
    try {
      const res = await fetch(`${apiBase}/api/fracture/predict`, {
        method: 'POST',
        body: formData,
        signal,
      });
      if (res.ok) {
        return await res.json();
      }
    } catch (err: any) {
      if (err.name === 'AbortError') throw err;
    }
  }

  // Realistic fallback demo response
  await new Promise((resolve) => setTimeout(resolve, 350));
  const name = file.name.toLowerCase();
  const isFracture = name.includes('fracture') || name.includes('hand') || name.includes('wrist');

  return {
    request_id: 'frac-' + Math.random().toString(36).substring(2, 9),
    status: 'success',
    anatomy: name.includes('hand') ? 'hand' : name.includes('wrist') ? 'wrist' : 'bone',
    finding: isFracture ? 'fracture_detected' : 'no_fracture_detected',
    abstained: false,
    fracture_probability: isFracture ? 0.892 : 0.084,
    calibrated_probability: isFracture ? 0.881 : 0.092,
    confidence_band: 'high',
    uncertainty_entropy: 0.052,
    quality: {
      acceptable: true,
      width: 512,
      height: 512,
      mean_intensity: 118.4,
      std_intensity: 54.2,
      issues: [],
      status: 'acceptable',
    },
    triage: {
      urgency: isFracture ? 'high' : 'routine',
      action: isFracture ? 'Orthopedic specialist consultation recommended' : 'Routine clinical follow-up',
      reason: isFracture ? 'Discontinuity in cortical bone margin identified.' : 'No cortical discontinuity detected.',
    },
    evidence: {
      gradcam_overlay_base64: null,
      disclaimer: 'Educational bone radiograph prototype fixture.',
    },
    explanation: isFracture
      ? 'Cortical surface irregularity and localized radiolucent fracture line detected in skeletal structures.'
      : 'Intact cortical bone contours with preserved joint spaces and alignment.',
    limitations: DEMO_LIMITATIONS,
    disclaimer: 'Educational prototype only.',
  };
}

export async function getFractureValidation(): Promise<any> {
  if (!isStaticHostWithoutBackend()) {
    const apiBase = getApiBase();
    try {
      const res = await fetch(`${apiBase}/api/fracture/validation`);
      if (res.ok) {
        return await handleResponse(res, 'Fracture validation report');
      }
    } catch {
      // Fall through
    }
  }

  return {
    status: 'available',
    report: {
      model_name: 'ConvNeXt-Base Fracture Model',
      dataset: 'Bone Fracture Radiographs',
      metrics: {
        accuracy: 0.912,
        auroc: 0.968,
        sensitivity: 0.924,
        specificity: 0.898,
        f1: 0.911,
      },
    },
  };
}

export async function getRegisteredModels(): Promise<any> {
  if (!isStaticHostWithoutBackend()) {
    const apiBase = getApiBase();
    try {
      const res = await fetch(`${apiBase}/api/models`);
      if (res.ok) {
        return await handleResponse(res, 'Registered models list');
      }
    } catch {
      // Fall through
    }
  }

  return {
    models: [
      {
        id: 'resnet18-chestmnist',
        name: 'MedGuard ResNet-18 (Chest Radiograph)',
        type: 'classification',
        target: 'pneumonia',
        status: 'ready',
      },
      {
        id: 'convnext-fracture',
        name: 'MedGuard ConvNeXt-Base (Bone Fracture)',
        type: 'classification',
        target: 'bone_fracture',
        status: 'ready',
      },
    ],
  };
}


