export interface Source {
    source: string;
    category: string;
    chunk: number;
    relevance_score: number;
}
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  timestamp: Date;
}

export interface AskRequest {
  question: string;
  conversation_id?: string;
  max_results?: number;
}

export interface AskResponse {
  question: string;
  answer: string;
  sources: Source[];
  conversation_id: string;
  response_time_ms: number;
  timestamp: string;
}