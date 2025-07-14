import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Card,
  CardContent,
  Paper,
  Chip,
  CircularProgress,
  Alert,
  Stack,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Rating,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Send as SendIcon,
  ExpandMore as ExpandMoreIcon,
  ContentCopy as CopyIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { apiService, type QueryRequest, type QueryResult, type AnswerSource } from '../services/api';

interface QueryInterfaceProps {
  sessionId?: string;
  onSessionUpdate?: (sessionId: string) => void;
}

interface StreamingState {
  isStreaming: boolean;
  content: string;
  error: string | null;
}

export const QueryInterface: React.FC<QueryInterfaceProps> = ({
  sessionId,
  onSessionUpdate,
}) => {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [streamingState, setStreamingState] = useState<StreamingState>({
    isStreaming: false,
    content: '',
    error: null,
  });
  const [showSources, setShowSources] = useState(true);
  
  const eventSourceRef = useRef<EventSource | null>(null);
  const pollingIntervalRef = useRef<number | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    const trimmedQuery = query.trim();
    setIsLoading(true);
    setResult(null);
    setStreamingState({ isStreaming: false, content: '', error: null });

    try {
      // Submit query
      const queryRequest: QueryRequest = {
        query: trimmedQuery,
        query_type: 'question',
        max_results: 5,
        include_sources: true,
        stream_response: true,
        session_id: sessionId,
      };

      const response = await apiService.submitQuery(queryRequest);
      
      // Update session if new one was created
      if (!sessionId && response.query_id) {
        onSessionUpdate?.(response.query_id);
      }

      // Start streaming
      startStreaming(response.query_id);
      
      // Start polling for result
      startPolling(response.query_id);

    } catch (error: any) {
      setIsLoading(false);
      setStreamingState({
        isStreaming: false,
        content: '',
        error: error.response?.data?.detail || error.message || 'Failed to submit query',
      });
    }
  };

  const startStreaming = (queryId: string) => {
    setStreamingState(prev => ({ ...prev, isStreaming: true, error: null }));
    
    try {
      eventSourceRef.current = apiService.createEventSource(queryId);
      
      eventSourceRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.chunk && data.chunk.content) {
            setStreamingState(prev => ({
              ...prev,
              content: prev.content + data.chunk.content,
            }));
          }
        } catch (err) {
          console.error('Error parsing stream data:', err);
        }
      };

      eventSourceRef.current.onerror = (error) => {
        console.error('EventSource error:', error);
        setStreamingState(prev => ({
          ...prev,
          isStreaming: false,
          error: 'Streaming connection failed',
        }));
        eventSourceRef.current?.close();
      };

    } catch (error) {
      console.error('Failed to start streaming:', error);
      setStreamingState(prev => ({
        ...prev,
        isStreaming: false,
        error: 'Failed to start streaming',
      }));
    }
  };

  const startPolling = (queryId: string) => {
    pollingIntervalRef.current = setInterval(async () => {
      try {
        const result = await apiService.getQueryStatus(queryId);
        
        if (result.status === 'completed' || result.status === 'failed') {
          setResult(result);
          setIsLoading(false);
          setStreamingState(prev => ({ ...prev, isStreaming: false }));
          
          // Close streaming and polling
          eventSourceRef.current?.close();
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 2000); // Poll every 2 seconds
  };

  const handleCopyAnswer = () => {
    if (result?.answer?.content) {
      navigator.clipboard.writeText(result.answer.content);
    }
  };

  const handleRetry = () => {
    if (query.trim()) {
      handleSubmit({ preventDefault: () => {} } as React.FormEvent);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  const renderSources = (sources: AnswerSource[]) => (
    <Accordion expanded={showSources} onChange={() => setShowSources(!showSources)}>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography variant="h6">
          Sources ({sources.length})
        </Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Stack spacing={2}>
          {sources.map((source, index) => (
            <Box key={index}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  {source.document_name}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {source.excerpt}
                </Typography>
                <Box display="flex" alignItems="center" gap={1} mt={1}>
                  <Chip 
                    label={`Relevance: ${(source.relevance_score * 100).toFixed(1)}%`} 
                    size="small" 
                    color="primary" 
                  />
                  {source.page_number && (
                    <Chip 
                      label={`Page ${source.page_number}`} 
                      size="small" 
                      variant="outlined" 
                    />
                  )}
                </Box>
              </Paper>
            </Box>
          ))}
        </Stack>
      </AccordionDetails>
    </Accordion>
  );

  return (
    <Box>
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Ask a Question
          </Typography>
          
          <form onSubmit={handleSubmit}>
            <Box display="flex" gap={2} alignItems="flex-start">
              <TextField
                fullWidth
                multiline
                minRows={2}
                maxRows={6}
                placeholder="Ask a question about your documents..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={isLoading}
                variant="outlined"
              />
              <Button
                type="submit"
                variant="contained"
                disabled={!query.trim() || isLoading}
                startIcon={isLoading ? <CircularProgress size={20} /> : <SendIcon />}
                sx={{ minWidth: 120, height: 56 }}
              >
                {isLoading ? 'Processing' : 'Ask'}
              </Button>
            </Box>
          </form>
        </CardContent>
      </Card>

      {/* Streaming Response */}
      {streamingState.isStreaming && streamingState.content && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box display="flex" alignItems="center" gap={1} mb={2}>
              <CircularProgress size={20} />
              <Typography variant="h6">Generating Answer...</Typography>
            </Box>
            <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
              {streamingState.content}
            </Typography>
          </CardContent>
        </Card>
      )}

      {/* Final Result */}
      {result && (
        <Card>
          <CardContent>
            <Box display="flex" justifyContent="between" alignItems="center" mb={2}>
              <Typography variant="h6">Answer</Typography>
              <Box display="flex" gap={1}>
                <Tooltip title="Copy answer">
                  <IconButton onClick={handleCopyAnswer} size="small">
                    <CopyIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Retry query">
                  <IconButton onClick={handleRetry} size="small">
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>

            {result.status === 'completed' && result.answer ? (
              <>
                <Typography variant="body1" sx={{ mb: 3, whiteSpace: 'pre-wrap' }}>
                  {result.answer.content}
                </Typography>

                {/* Answer Quality Metrics */}
                <Box mb={3}>
                  <Typography variant="subtitle2" gutterBottom>
                    Answer Quality
                  </Typography>
                  <Box display="flex" alignItems="center" gap={2}>
                    <Rating
                      value={result.answer.confidence_score * 5}
                      readOnly
                      precision={0.1}
                      size="small"
                    />
                    <Typography variant="body2" color="text.secondary">
                      Confidence: {(result.answer.confidence_score * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Generated in: {result.answer.generation_time.toFixed(2)}s
                    </Typography>
                  </Box>
                </Box>

                {/* Sources */}
                {result.answer.sources.length > 0 && renderSources(result.answer.sources)}
              </>
            ) : result.status === 'failed' ? (
              <Alert severity="error">
                {result.error_message || 'Query processing failed'}
              </Alert>
            ) : (
              <Box display="flex" alignItems="center" gap={2}>
                <CircularProgress size={20} />
                <Typography>Processing query...</Typography>
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* Streaming Error */}
      {streamingState.error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {streamingState.error}
        </Alert>
      )}
    </Box>
  );
};