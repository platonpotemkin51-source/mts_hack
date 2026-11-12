// Contract v1. Planned API; only /api/health exists in the initial scaffold.
export interface KnowledgeChunk {
  chunk_id: string;
  source: string;
  title: string;
  text: string;
  score: number;
}

export interface ClientBrief {
  object: string | null;
  problem: string | null;
  known_parameters: string[];
  recommended_products: string[];
  missing_fields: string[];
}

export interface Timings {
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
}

export interface ChatRequest {
  session_id: string | null;
  message: string;
  debug?: boolean;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  sources: KnowledgeChunk[];
  brief: ClientBrief;
  lead_offer: boolean;
  debug: Timings | null;
}

export interface SessionSummary {
  session_id: string;
  status: "active" | "closed";
  created_at: string;
  updated_at: string;
}

export interface SessionDetail extends SessionSummary {
  messages: Array<{
    role: "user" | "assistant";
    content: string;
    created_at: string;
  }>;
  brief: ClientBrief;
  lead_id: string | null;
}

export interface LeadRequest {
  session_id: string;
  full_name: string;
  phone: string;
  company: string;
  consent: true;
}

export interface LeadResponse {
  lead_id: string;
  session_id: string;
  summary: string;
  created_at: string;
}

export interface ApiError {
  error: { code: string; message: string };
}
