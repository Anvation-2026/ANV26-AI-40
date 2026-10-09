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

async function handleResponse<T>(res: Response, endpointDesc: string): Promise<T> {
  const contentType = res.headers.get('content-type') || '';
  if (!contentType.includes('application/json')) {
    // Likely a Vite proxy connection error (ECONNREFUSED) which returns HTML/plain text 500
    throw new NetworkError('API server offline — connect to backend on port 8000.');
  }

  const data = await res.json();
  if (!res.ok) {
    const errorMsg = data?.message || data?.detail || `${endpointDesc} returned status ${res.status}`;
    throw new NetworkError(errorMsg);
  }
  return data as T;
}

export async function getHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    return await handleResponse<HealthResponse>(res, 'Health check');
  } catch (err: any) {
    if (err instanceof NetworkError) throw err;
    throw new NetworkError('API server offline — connect to backend on port 8000.');
  }
}

export async function getModelStatus(): Promise<ModelStatusResponse> {
  try {
    const res = await fetch(`${API_BASE}/api/model/status`);
    return await handleResponse<ModelStatusResponse>(res, 'Model status');
  } catch (err: any) {
    if (err instanceof NetworkError) throw err;
    throw new NetworkError('API server offline — connect to backend on port 8000.');
  }
}

export async function getValidation(): Promise<ValidationResponse> {
  try {
    const res = await fetch(`${API_BASE}/api/validation`);
    return await handleResponse<ValidationResponse>(res, 'Validation report');
  } catch (err: any) {
    if (err instanceof NetworkError) throw err;
    throw new NetworkError('API server offline — connect to backend on port 8000.');
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
    throw new NetworkError('API server offline — failed to reach analysis endpoint on port 8000.');
  }

  // Parse JSON response. Note: all responses (200, 400, 413, 415, 500, 503) return AnalysisResponse structure
  let data: any;
  try {
    data = await res.json();
  } catch {
    throw new NetworkError('API server offline or invalid response from port 8000.');
  }

  return data as AnalysisResponse;
}

export async function getFractureStatus(): Promise<any> {
  try {
    const res = await fetch(`${API_BASE}/api/fracture/status`);
    return await handleResponse(res, 'Fracture model status');
  } catch (err: any) {
    if (err instanceof NetworkError) throw err;
    throw new NetworkError('API server offline — connect to backend on port 8000.');
  }
}

export async function predictFracture(
  file: File,
  signal?: AbortSignal
): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/fracture/predict`, {
      method: 'POST',
      body: formData,
      signal,
    });
  } catch (err: any) {
    if (err.name === 'AbortError') {
      throw err;
    }
    throw new NetworkError('API server offline — failed to reach fracture analysis endpoint.');
  }

  let data: any;
  try {
    data = await res.json();
  } catch {
    throw new NetworkError('API server offline or invalid response from fracture endpoint.');
  }

  return data;
}

export async function getFractureValidation(): Promise<any> {
  try {
    const res = await fetch(`${API_BASE}/api/fracture/validation`);
    return await handleResponse(res, 'Fracture validation report');
  } catch (err: any) {
    if (err instanceof NetworkError) throw err;
    throw new NetworkError('API server offline — connect to backend on port 8000.');
  }
}

export async function getRegisteredModels(): Promise<any> {
  try {
    const res = await fetch(`${API_BASE}/api/models`);
    return await handleResponse(res, 'Registered models list');
  } catch (err: any) {
    if (err instanceof NetworkError) throw err;
    throw new NetworkError('API server offline — connect to backend on port 8000.');
  }
}
