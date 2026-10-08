import { AnalysisResponse } from '../types/analysis';

export interface RecentAnalysisItem {
  id: string;
  timestamp: string;
  filename: string;
  previewUrl?: string;
  finding: string | null;
  status: string;
  probability: number | null;
  uncertaintyLevel: string;
  result: AnalysisResponse;
}

const STORAGE_KEY = 'medguard_recent_analyses_v1';
const MAX_HISTORY = 8;

export function getRecentAnalyses(): RecentAnalysisItem[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

export function saveRecentAnalysis(
  filename: string,
  result: AnalysisResponse,
  previewDataUrl?: string
): void {
  try {
    const current = getRecentAnalyses();
    const newItem: RecentAnalysisItem = {
      id: result.request_id || `hist-${Date.now()}`,
      timestamp: new Date().toISOString(),
      filename,
      previewUrl: previewDataUrl,
      finding: result.finding,
      status: result.status,
      probability: result.probability,
      uncertaintyLevel: result.uncertainty?.level || 'unknown',
      result,
    };

    // Filter out duplicate if same ID
    const filtered = current.filter((item) => item.id !== newItem.id);
    const updated = [newItem, ...filtered].slice(0, MAX_HISTORY);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  } catch (e) {
    console.warn('Could not save analysis history:', e);
  }
}

export function clearRecentAnalyses(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (e) {
    console.warn('Could not clear analysis history:', e);
  }
}
