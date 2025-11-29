export const API_URL = import.meta.env.PROD ? "" : "http://localhost:8000";
export const WS_URL = import.meta.env.PROD 
  ? ((window.location.protocol === "https:") ? "wss://" : "ws://") + window.location.host + "/ws/logs"
  : "ws://localhost:8000/ws/logs";

export interface ImageData {
  id: string;
  data: string;
}

export interface ImagePair {
  image1: ImageData;
  image2: ImageData;
  debug_strategy: string;
}

export interface VotePayload {
  id1: string;
  id2: string;
  are_same: boolean;
}

export async function fetchPair(): Promise<ImagePair> {
  const response = await fetch(`${API_URL}/pair`);
  if (!response.ok) throw new Error("Failed to fetch pair");
  return response.json();
}

export async function sendVote(payload: VotePayload): Promise<void> {
  await fetch(`${API_URL}/vote`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
