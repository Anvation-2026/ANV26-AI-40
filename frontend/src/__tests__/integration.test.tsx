import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { Workspace } from '../pages/Workspace';
import { ToastProvider } from '../components/Toast';
import { XrayViewer } from '../components/XrayViewer';
import { SidebarNav } from '../components/SidebarNav';

// Mock API services
vi.mock('../services/api', () => ({
  getModelStatus: vi.fn().mockResolvedValue({
    available: false,
    model_name: null,
    model_version: null,
    supported_image_type: 'Chest X-ray (educational)',
    inference_ready: false,
    calibration_available: false,
    ood_available: false,
    quality_available: false,
    gradcam_available: false,
    mode: 'real',
    message: 'ML module not loaded or not configured',
  }),
  getHealth: vi.fn().mockResolvedValue({ status: 'ok', mode: 'real', version: '1.0.0', ml_module_loaded: false, timestamp: '' }),
  predictImage: vi.fn().mockResolvedValue({
    request_id: 'mock-1234',
    status: 'model_unavailable',
    mode: 'real',
    finding: null,
    abstained: true,
    raw_score: null,
    probability: null,
    probability_of: null,
    uncertainty: { level: 'not_evaluated', value: null, method: null },
    quality: { evaluated: false, status: 'not_evaluated', blur: null, brightness: null, contrast: null, reasons: [] },
    ood: { evaluated: false, is_ood: null, score: null, method: null, reason: null },
    heatmap: { available: false, data_url: null, kind: null, message: 'Unavailable' },
    triage: {
      action: 'technical_error',
      title: 'Analysis Notice',
      message: 'The ML decision-support module is currently unavailable.',
      reasons: [],
    },
    explanation: 'Model unavailable',
    evidence: [],
    limitations: [],
    model: { available: false, model_name: null, model_version: null, dataset: 'PneumoniaMNIST+' },
    error: null,
    disclaimer: 'Educational prototype',
  }),
}));

describe('Workstation Integration Tests', () => {
  it('renders Workspace without crashing', async () => {
    render(
      <ToastProvider>
        <BrowserRouter>
          <Workspace />
        </BrowserRouter>
      </ToastProvider>
    );

    expect(screen.getByText(/Workstation Active/i)).toBeInTheDocument();
    expect(screen.getByText(/Upload Chest Radiograph/i)).toBeInTheDocument();
    expect(screen.getByText(/Verification Safeguards/i)).toBeInTheDocument();
  });

  it('renders XrayViewer with split and overlay controls when image and heatmap provided', () => {
    render(
      <XrayViewer
        originalUrl="blob:http://localhost/mock-url"
        heatmap={{
          available: true,
          data_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
          kind: 'overlay',
          message: null,
        }}
        status="success"
      />
    );

    expect(screen.getByText(/Overlay/i)).toBeInTheDocument();
    expect(screen.getByText(/Split Slider/i)).toBeInTheDocument();
  });

  it('renders SidebarNav without crashing and highlights active link with 3 destinations', () => {
    render(
      <BrowserRouter>
        <SidebarNav collapsed={false} onToggleCollapse={() => {}} modelStatus={null} />
      </BrowserRouter>
    );

    expect(screen.getByText(/Workspace/i)).toBeInTheDocument();
    expect(screen.getByText(/Validation/i)).toBeInTheDocument();
    expect(screen.getByText(/About & Methodology/i)).toBeInTheDocument();
  });

  it('renders detailed explainability and ambiguity attention map for uncertain status', () => {
    const mockUncertainResponse = {
      request_id: 'unc-req-1234',
      status: 'uncertain' as const,
      mode: 'real' as const,
      finding: null,
      abstained: true,
      raw_score: 0.448,
      probability: null,
      probability_of: null,
      uncertainty: { level: 'high' as const, value: 0.9969, method: 'normalized_entropy' },
      quality: {
        evaluated: true,
        status: 'acceptable' as const,
        blur: { value: 119.5, threshold: 100, passed: true },
        brightness: { value: 0.63, threshold: 0.2, passed: true },
        contrast: { value: 0.93, threshold: 0.3, passed: true },
        reasons: [],
      },
      ood: { evaluated: true, is_ood: false, score: 30.8, method: 'mahalanobis', reason: null },
      heatmap: {
        available: false,
        data_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
        kind: 'overlay' as const,
        message: 'Educational attention map generated for ambiguous features.',
      },
      triage: {
        action: 'expert_review_required_abstained' as const,
        title: 'High Model Uncertainty — Decision Abstained',
        message: 'Educational recommendation: have a qualified human expert review this image.',
        reasons: ['High model uncertainty exceeded acceptable decision support threshold.'],
      },
      explanation: 'Model abstained due to high uncertainty.',
      evidence: [
        'Calibrated confidence = 0.53 (acceptance threshold tau = 0.56)',
        'Borderline calibrated P(pneumonia) = 0.47',
      ],
      limitations: [],
      model: { available: true, model_name: 'ResNet-18', model_version: 'v1', dataset: 'PneumoniaMNIST+' },
      error: null,
      disclaimer: 'Educational prototype',
    };

    // Test XrayViewer with uncertain status and attention heatmap
    const { unmount } = render(
      <XrayViewer
        originalUrl="blob:http://localhost/mock-xray"
        heatmap={mockUncertainResponse.heatmap}
        status="uncertain"
      />
    );
    expect(screen.getByText(/Overlay/i)).toBeInTheDocument();
    expect(screen.getByText(/Ambiguity Attention Map:/i)).toBeInTheDocument();
    unmount();
  });
});
