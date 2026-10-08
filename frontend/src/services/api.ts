import {
  AnalysisResponse,
  HealthResponse,
  ModelStatusResponse,
  ValidationResponse,
} from '../types/analysis';

const API_BASE = import.meta.env.VITE_API_BASE || '';

export class NetworkError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'NetworkError';
  }
}

export async function getHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) {
      throw new NetworkError(`Health check failed with HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err: any) {
    if (err instanceof NetworkError) throw err;
    throw new NetworkError('Backend unreachable — please verify the API server is running.');
  }
}

export async function getModelStatus(): Promise<ModelStatusResponse> {
  try {
    const res = await fetch(`${API_BASE}/api/model/status`);
    if (!res.ok) {
      throw new NetworkError(`Model status failed with HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err: any) {
    if (err instanceof NetworkError) throw err;
    throw new NetworkError('Backend unreachable — could not retrieve model status.');
  }
}

export async function getValidation(): Promise<ValidationResponse> {
  try {
    const res = await fetch(`${API_BASE}/api/validation`);
    if (!res.ok) {
      throw new NetworkError(`Validation fetch failed with HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err: any) {
    if (err instanceof NetworkError) throw err;
    throw new NetworkError('Backend unreachable — could not load validation metrics.');
  }
}

export async function predictImage(
  file: File,
  signal?: AbortSignal,
  demoScenario?: string
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const headers: Record<string, string> = {};
  if (demoScenario) {
    headers['X-Demo-Scenario'] = demoScenario;
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      body: formData,
      headers,
      signal,
    });
  } catch (err: any) {
    if (err.name === 'AbortError') {
      throw err;
    }
    throw new NetworkError('Backend unreachable — failed to connect to analysis server.');
  }

  // Parse JSON response. Note: all responses (200, 400, 413, 415, 500, 503) return AnalysisResponse structure
  let data: any;
  try {
    data = await res.json();
  } catch {
    throw new NetworkError(`Server responded with HTTP ${res.status} but returned non-JSON content.`);
  }

  return data as AnalysisResponse;
}
