import { AskRequest, AskResponse } from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/** Max question length accepted by the backend (AskRequest.question max_length). */
export const MAX_QUESTION_LENGTH = 500;

/**
 * Error from the API.
 * status is 0 when the server couldn't be reached at all (network error).
 */
export class APIError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'APIError';
  }
}

/** Turn an APIError into a message suitable for the chat window. */
export function describeError(err: unknown): string {
  if (!(err instanceof APIError)) {
    return 'Sorry, something unexpected went wrong. Please try again.';
  }
  if (err.status === 0) {
    return `I can't reach the AskMyCity server. Please make sure the API is running at ${API_BASE_URL}.`;
  }
  if (err.status === 429) {
    return "You're sending questions too quickly. Please wait a minute and try again.";
  }
  if (err.status === 400 || err.status === 422) {
    return `I couldn't accept that question. Please keep it under ${MAX_QUESTION_LENGTH} characters and try again.`;
  }
  return 'Sorry, the server had a problem answering that question. Please try again in a moment.';
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    // fetch only rejects on network failure (server down, CORS, offline)
    throw new APIError(0, 'Network error');
  }

  if (!response.ok) {
    throw new APIError(response.status, `API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

export class APIClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async ask(body: AskRequest): Promise<AskResponse> {
    return request<AskResponse>(`${this.baseURL}/api/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
  }

  async getHealth(): Promise<unknown> {
    return request(`${this.baseURL}/health`);
  }

  async getStats(): Promise<unknown> {
    return request(`${this.baseURL}/stats`);
  }
}

export const apiClient = new APIClient();
