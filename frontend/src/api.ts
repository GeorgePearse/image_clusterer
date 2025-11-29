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

export async function fetchPoints(): Promise<Point[]> {
  const response = await fetch(`${API_URL}/points`);
  if (!response.ok) throw new Error("Failed to fetch points");
  return response.json();
}
