import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Typography,
  Chip,
  TextField,
  Alert,
  LinearProgress,
  Card,
  CardContent,
  Paper,
  Stack,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  Check as CheckIcon,
} from '@mui/icons-material';
import { apiService, type Document } from '../services/api';

interface DocumentUploadProps {
  onUploadComplete?: (document: Document) => void;
  onUploadError?: (error: string) => void;
}

interface UploadState {
  uploading: boolean;
  progress: number;
  error: string | null;
  success: boolean;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onUploadComplete,
  onUploadError,
}) => {
  const [tags, setTags] = useState<string>('');
  const [uploadState, setUploadState] = useState<UploadState>({
    uploading: false,
    progress: 0,
    error: null,
    success: false,
  });

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    
    // Validate file type
    const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
    if (!allowedTypes.includes(file.type)) {
      const error = 'Only PDF, DOCX, and TXT files are supported';
      setUploadState(prev => ({ ...prev, error }));
      onUploadError?.(error);
      return;
    }

    // Validate file size (50MB limit)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      const error = 'File size must be less than 50MB';
      setUploadState(prev => ({ ...prev, error }));
      onUploadError?.(error);
      return;
    }

    setUploadState({
      uploading: true,
      progress: 0,
      error: null,
      success: false,
    });

    try {
      // Parse tags
      const tagList = tags
        .split(',')
        .map(tag => tag.trim())
        .filter(tag => tag.length > 0);

      // Upload document
      const document = await apiService.uploadDocument(file, tagList);
      
      setUploadState({
        uploading: false,
        progress: 100,
        error: null,
        success: true,
      });

      onUploadComplete?.(document);
      
      // Reset form after success
      setTimeout(() => {
        setUploadState(prev => ({ ...prev, success: false }));
        setTags('');
      }, 3000);

    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message || 'Upload failed';
      setUploadState({
        uploading: false,
        progress: 0,
        error: errorMessage,
        success: false,
      });
      onUploadError?.(errorMessage);
    }
  }, [tags, onUploadComplete, onUploadError]);

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragReject,
  } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
    },
    maxFiles: 1,
    disabled: uploadState.uploading,
  });

  const handleTagKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      event.preventDefault();
    }
  };

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Upload Document
        </Typography>
        
        <Stack spacing={3}>
          <Box>
            <Paper
              {...getRootProps()}
              sx={{
                p: 3,
                border: '2px dashed',
                borderColor: isDragActive 
                  ? 'primary.main' 
                  : isDragReject 
                  ? 'error.main' 
                  : 'grey.300',
                backgroundColor: isDragActive 
                  ? 'primary.light' 
                  : isDragReject 
                  ? 'error.light' 
                  : 'grey.50',
                cursor: uploadState.uploading ? 'not-allowed' : 'pointer',
                textAlign: 'center',
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  backgroundColor: uploadState.uploading ? 'grey.50' : 'grey.100',
                },
              }}
            >
              <input {...getInputProps()} />
              
              <Box display="flex" flexDirection="column" alignItems="center" gap={2}>
                {uploadState.uploading ? (
                  <>
                    <LinearProgress sx={{ width: '100%', mb: 2 }} />
                    <Typography variant="body1">Uploading...</Typography>
                  </>
                ) : uploadState.success ? (
                  <>
                    <CheckIcon color="success" sx={{ fontSize: 48 }} />
                    <Typography variant="body1" color="success.main">
                      Document uploaded successfully!
                    </Typography>
                  </>
                ) : (
                  <>
                    <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary' }} />
                    <Typography variant="h6">
                      {isDragActive
                        ? 'Drop your document here...'
                        : 'Drag & drop a document, or click to browse'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Supports PDF, DOCX, and TXT files (max 50MB)
                    </Typography>
                  </>
                )}
              </Box>
            </Paper>
          </Box>

          <Box>
            <TextField
              fullWidth
              label="Tags (comma-separated)"
              placeholder="Enter tags to help organize your document..."
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              onKeyPress={handleTagKeyPress}
              disabled={uploadState.uploading}
              helperText="Example: research, meeting-notes, legal-document"
            />
          </Box>

          {tags && (
            <Box>
              <Box display="flex" flexWrap="wrap" gap={1}>
                {tags.split(',').map((tag, index) => {
                  const trimmedTag = tag.trim();
                  return trimmedTag ? (
                    <Chip key={index} label={trimmedTag} size="small" />
                  ) : null;
                })}
              </Box>
            </Box>
          )}

          {uploadState.error && (
            <Box>
              <Alert severity="error" sx={{ mt: 2 }}>
                {uploadState.error}
              </Alert>
            </Box>
          )}

          {uploadState.success && (
            <Box>
              <Alert severity="success" sx={{ mt: 2 }}>
                Document uploaded successfully! It will be processed and available for queries shortly.
              </Alert>
            </Box>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};