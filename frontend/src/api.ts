export const API_URL = import.meta.env.PROD ? "" : "http://localhost:8000";
export const WS_URL = import.meta.env.PROD 
  ? ((window.location.protocol === "https:") ? "wss://" : "ws://") + window.location.host + "/ws/logs"
  : "ws://localhost:8000/ws/logs";

export interface ImageData {
  id: string;
  data: string;
}

export interface NextSampleResponse {
  image?: ImageData;
  suggestion: string | null;
  debug_info?: any;
  status?: string;
  message?: string;
}

export interface LabelPayload {
  image_id: string;
  label: string;
}

export interface Point {
  id: string;
  x: number;
  y: number;
  label?: string | null;
  predicted_label?: string | null;
  confidence?: number;
}

export async function fetchNextSample(): Promise<NextSampleResponse> {
  const response = await fetch(`${API_URL}/next`);
  if (!response.ok) throw new Error("Failed to fetch next sample");
  return response.json();
}

export async function sendLabel(payload: LabelPayload): Promise<void> {
  await fetch(`${API_URL}/label`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function fetchPoints(includePredictions: boolean = false): Promise<Point[]> {
  const url = includePredictions ? `${API_URL}/points?predictions=true` : `${API_URL}/points`;
  const response = await fetch(url);
  if (!response.ok) throw new Error("Failed to fetch points");
  return response.json();
}

export interface StatusResponse {
  ready: boolean;
  stage: string;
  progress: number;
  message: string;
}

export async function fetchStatus(): Promise<StatusResponse> {
  const response = await fetch(`${API_URL}/status`);
  if (!response.ok) throw new Error("Failed to fetch status");
  return response.json();
}

export interface KnnConfig {
  k_neighbors: number;
  min_k: number;
  max_k: number;
}

export async function fetchKnnConfig(): Promise<KnnConfig> {
  const response = await fetch(`${API_URL}/config/knn`);
  if (!response.ok) throw new Error("Failed to fetch KNN config");
  return response.json();
}

export async function setKnnConfig(k: number): Promise<KnnConfig> {
  const response = await fetch(`${API_URL}/config/knn?k=${k}`, {
    method: "POST",
  });
  if (!response.ok) throw new Error("Failed to set KNN config");
  return response.json();
}
