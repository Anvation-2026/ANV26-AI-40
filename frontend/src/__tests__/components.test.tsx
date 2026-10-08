import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { FindingCard } from '../components/FindingCard';
import { HeatmapViewer } from '../components/HeatmapViewer';
import { MetricCard } from '../components/MetricCard';
import { ImageUploader } from '../components/ImageUploader';
import { AnalysisResponse } from '../types/analysis';

const baseMockResponse: AnalysisResponse = {
  request_id: 'test-uuid-1234',
  status: 'success',
  mode: 'real',
  finding: 'pneumonia',
  abstained: false,
  raw_score: 0.884,
  probability: 0.87,
  probability_of: 'pneumonia',
  uncertainty: { level: 'low', value: 0.042, method: 'MC-Dropout' },
  quality: {
    evaluated: true,
    status: 'acceptable',
    blur: { value: 240, threshold: 100, passed: true },
    brightness: { value: 120, threshold: 40, passed: true },
    contrast: { value: 60, threshold: 20, passed: true },
    reasons: [],
  },
  ood: { evaluated: true, is_ood: false, score: 0.1, method: 'Mahalanobis', reason: null },
  heatmap: { available: true, data_url: 'data:image/png;base64,iVBOR', kind: 'overlay', message: null },
  triage: {
    action: 'expert_review_recommended',
    title: 'Educational Review Recommended',
    message: 'Pattern match detected for pneumonia.',
    reasons: ['Pattern match detected'],
  },
  explanation: 'Model predicted pneumonia with 87% probability.',
  evidence: ['Good quality'],
  limitations: ['Educational only'],
  model: { available: true, model_name: 'ResNet-18', model_version: 'v1', dataset: 'PneumoniaMNIST+' },
  error: null,
  disclaimer: 'Educational research prototype.',
};

describe('FindingCard Component', () => {
  it('renders finding and educational label for success state', () => {
    render(<FindingCard analysis={baseMockResponse} />);
    expect(screen.getByText(/Pneumonia Pattern/i)).toBeInTheDocument();
    expect(screen.getByText(/Model Finding \(Educational\)/i)).toBeInTheDocument();
  });

  it('renders NO finding text and shows abstention for uncertain state', () => {
    const uncertainMock: AnalysisResponse = {
      ...baseMockResponse,
      status: 'uncertain',
      finding: null,
      probability: null,
      triage: {
        action: 'expert_review_required_abstained',
        title: 'Decision Abstained Due to Uncertainty',
        message: 'High uncertainty detected. Human review required.',
        reasons: ['Uncertainty above threshold'],
      },
    };
    render(<FindingCard analysis={uncertainMock} />);
    expect(screen.queryByText(/Pneumonia Pattern/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Normal \/ Unremarkable/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Decision Abstained Due to Uncertainty/i)).toBeInTheDocument();
    expect(screen.getByText(/No definitive finding or probability is displayed/i)).toBeInTheDocument();
  });

  it('renders NO finding text for poor quality state', () => {
    const poorMock: AnalysisResponse = {
      ...baseMockResponse,
      status: 'poor_quality',
      finding: null,
      probability: null,
      triage: {
        action: 'resubmit_better_sample',
        title: 'Poor Quality Image',
        message: 'Severe blur detected.',
        reasons: ['Motion blur'],
      },
    };
    render(<FindingCard analysis={poorMock} />);
    expect(screen.queryByText(/Pneumonia Pattern/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Poor Quality Image/i)).toBeInTheDocument();
  });

  it('renders NO finding text for OOD state', () => {
    const oodMock: AnalysisResponse = {
      ...baseMockResponse,
      status: 'ood',
      finding: null,
      probability: null,
      triage: {
        action: 'unsupported_input',
        title: 'Out of Distribution',
        message: 'Non-chest radiograph detected.',
        reasons: ['OOD score exceeded'],
      },
    };
    render(<FindingCard analysis={oodMock} />);
    expect(screen.queryByText(/Pneumonia Pattern/i)).not.toBeInTheDocument();
    expect(screen.getAllByText(/Out of Distribution/i).length).toBeGreaterThan(0);
  });
});

describe('HeatmapViewer Component', () => {
  it('shows fallback card when heatmap is unavailable', () => {
    render(
      <HeatmapViewer
        heatmap={{ available: false, data_url: null, kind: null, message: 'Grad-CAM calculation skipped' }}
        status="success"
      />
    );
    expect(screen.getByText(/Grad-CAM calculation skipped/i)).toBeInTheDocument();
  });

  it('shows suppressed notice for abstained or rejected states', () => {
    render(
      <HeatmapViewer
        heatmap={{ available: false, data_url: null, kind: null, message: null }}
        status="uncertain"
      />
    );
    expect(screen.getByText(/No heatmap is shown for rejected or abstained inputs/i)).toBeInTheDocument();
  });
});

describe('MetricCard Component', () => {
  it('renders formatted percentage when value is provided', () => {
    render(<MetricCard label="Sensitivity" value={0.915} isPercentage={true} />);
    expect(screen.getByText(/Sensitivity/i)).toBeInTheDocument();
    expect(screen.getByText('91.5%')).toBeInTheDocument();
  });

  it('renders "Not evaluated" when value is null or undefined', () => {
    render(<MetricCard label="Specificity" value={null} />);
    expect(screen.getByText(/Specificity/i)).toBeInTheDocument();
    expect(screen.getByText(/Not evaluated/i)).toBeInTheDocument();
  });
});

describe('ImageUploader Component', () => {
  it('rejects unsupported file formats with user-friendly error', () => {
    const onSelect = vi.fn();
    const { container } = render(<ImageUploader onFileSelected={onSelect} />);

    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const txtFile = new File(['dummy content'], 'document.txt', { type: 'text/plain' });

    fireEvent.change(input, { target: { files: [txtFile] } });

    expect(screen.getByText(/Please select a valid image file/i)).toBeInTheDocument();
    expect(onSelect).not.toHaveBeenCalled();
  });

  it('rejects oversized files exceeding 10 MB', () => {
    const onSelect = vi.fn();
    const { container } = render(<ImageUploader onFileSelected={onSelect} />);

    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    // Mock oversized file > 10MB
    const bigFile = new File([''], 'huge.png', { type: 'image/png' });
    Object.defineProperty(bigFile, 'size', { value: 11 * 1024 * 1024 });

    fireEvent.change(input, { target: { files: [bigFile] } });

    expect(screen.getByText(/Image size exceeds the 10 MB maximum limit/i)).toBeInTheDocument();
    expect(onSelect).not.toHaveBeenCalled();
  });
});
