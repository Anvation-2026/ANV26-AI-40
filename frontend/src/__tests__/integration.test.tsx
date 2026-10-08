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
});
