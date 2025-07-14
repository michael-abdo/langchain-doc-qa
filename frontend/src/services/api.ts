import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Document {
  id: string;
  filename: string;
  original_filename: string;
  file_type: string;
  file_size_mb: number;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed';
  processing_error?: string;
  content_preview?: string;
  total_chunks?: number;
  created_at: string;
  tags?: string[];
}

export interface QueryRequest {
  query: string;
  query_type?: 'question' | 'search' | 'summary' | 'chat';
  context_documents?: string[];
  max_results?: number;
  include_sources?: boolean;
  stream_response?: boolean;
  session_id?: string;
}

export interface QueryResponse {
  query_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  estimated_completion_time?: number;
  message: string;
  created_at: string;
}

export interface AnswerSource {
  document_id: string;
  document_name: string;
  chunk_id: string;
  relevance_score: number;
  page_number?: number;
  excerpt: string;
}

export interface Answer {
  content: string;
  confidence_score: number;
  sources: AnswerSource[];
  generation_time: number;
  model_used: string;
  token_usage: Record<string, number>;
}

export interface QueryResult {
  query_id: string;
  query: string;
  query_type: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  answer?: Answer;
  error_message?: string;
  processing_time: number;
  created_at: string;
  completed_at?: string;
  session_id?: string;
}

export class ApiService {
  // Document management
  async uploadDocument(file: File, tags?: string[]): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file);
    if (tags && tags.length > 0) {
      formData.append('tags', tags.join(','));
    }

    const response = await api.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async getDocuments(params?: {
    page?: number;
    per_page?: number;
    file_type?: string;
    processing_status?: string;
    tags?: string;
  }): Promise<{ documents: Document[]; total: number; page: number; per_page: number }> {
    const response = await api.get('/documents', { params });
    return response.data;
  }

  async getDocument(documentId: string, includeChunks = false): Promise<Document> {
    const response = await api.get(`/documents/${documentId}`, {
      params: { include_chunks: includeChunks },
    });
    return response.data;
  }

  async deleteDocument(documentId: string, hardDelete = false): Promise<void> {
    await api.delete(`/documents/${documentId}`, {
      params: { hard_delete: hardDelete },
    });
  }

  async getDocumentStatus(documentId: string): Promise<{
    document_id: string;
    processing_status: string;
    processing_error?: string;
    total_chunks?: number;
    progress_percentage?: number;
  }> {
    const response = await api.get(`/documents/${documentId}/status`);
    return response.data;
  }

  // Query management
  async submitQuery(queryRequest: QueryRequest): Promise<QueryResponse> {
    const response = await api.post('/query', queryRequest);
    return response.data;
  }

  async getQueryStatus(queryId: string): Promise<QueryResult> {
    const response = await api.get(`/query/${queryId}`);
    return response.data;
  }

  async getQueries(params?: {
    session_id?: string;
    status?: string;
    page?: number;
    per_page?: number;
  }): Promise<{ queries: QueryResult[]; total: number; page: number; per_page: number }> {
    const response = await api.get('/queries', { params });
    return response.data;
  }

  async cancelQuery(queryId: string): Promise<void> {
    await api.delete(`/query/${queryId}`);
  }

  // Chat functionality
  async sendChatMessage(message: string, sessionId?: string): Promise<{
    session_id: string;
    message_id: string;
    response: string;
    sources: AnswerSource[];
    confidence_score: number;
    processing_time: number;
  }> {
    const response = await api.post('/chat', {
      message,
      session_id: sessionId,
      stream_response: false,
    });
    return response.data;
  }

  async getChatHistory(sessionId: string): Promise<{
    session: {
      session_id: string;
      created_at: string;
      last_activity: string;
      message_count: number;
    };
    messages: Array<{
      role: 'user' | 'assistant';
      content: string;
      timestamp: string;
    }>;
  }> {
    const response = await api.get(`/chat/${sessionId}/history`);
    return response.data;
  }

  // Streaming support
  createEventSource(queryId: string): EventSource {
    return new EventSource(`${API_BASE_URL}/query/${queryId}/stream`);
  }

  // Health check
  async getHealth(): Promise<{ status: string; details: Record<string, any> }> {
    const response = await api.get('/health/live');
    return response.data;
  }
}

export const apiService = new ApiService();