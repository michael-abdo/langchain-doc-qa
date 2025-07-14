import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  IconButton,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Pagination,
  TextField,
  MenuItem,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Description as DocumentIcon,
  CheckCircle as CompleteIcon,
  Error as ErrorIcon,
  Schedule as PendingIcon,
  Autorenew as ProcessingIcon,
} from '@mui/icons-material';
import { apiService, type Document } from '../services/api';

interface DocumentListProps {
  refreshTrigger?: number;
  onDocumentSelect?: (document: Document) => void;
}

export const DocumentList: React.FC<DocumentListProps> = ({
  refreshTrigger,
  onDocumentSelect,
}) => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [filters, setFilters] = useState({
    file_type: '',
    processing_status: '',
    tags: '',
  });
  const [deleteDialog, setDeleteDialog] = useState<{
    open: boolean;
    document: Document | null;
  }>({ open: false, document: null });

  const perPage = 10;

  const loadDocuments = async () => {
    try {
      setLoading(true);
      setError(null);

      const params = {
        page,
        per_page: perPage,
        ...(filters.file_type && { file_type: filters.file_type }),
        ...(filters.processing_status && { processing_status: filters.processing_status }),
        ...(filters.tags && { tags: filters.tags }),
      };

      const response = await apiService.getDocuments(params);
      setDocuments(response.documents);
      setTotal(response.total);
      setTotalPages(Math.ceil(response.total / perPage));
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, [page, filters, refreshTrigger]);

  const handleDeleteDocument = async (document: Document) => {
    try {
      await apiService.deleteDocument(document.id);
      setDeleteDialog({ open: false, document: null });
      loadDocuments();
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to delete document');
    }
  };

  const handleFilterChange = (field: string, value: string) => {
    setFilters(prev => ({ ...prev, [field]: value }));
    setPage(1); // Reset to first page when filtering
  };

  const formatFileSize = (sizeInMB: number): string => {
    if (sizeInMB < 1) {
      return `${(sizeInMB * 1024).toFixed(1)} KB`;
    }
    return `${sizeInMB.toFixed(1)} MB`;
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleString();
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CompleteIcon color="success" />;
      case 'processing':
        return <ProcessingIcon color="info" />;
      case 'failed':
        return <ErrorIcon color="error" />;
      case 'pending':
      default:
        return <PendingIcon color="warning" />;
    }
  };

  const getStatusColor = (status: string): "default" | "primary" | "secondary" | "error" | "info" | "success" | "warning" => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'info';
      case 'failed':
        return 'error';
      case 'pending':
      default:
        return 'warning';
    }
  };

  if (loading && documents.length === 0) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" gap={2}>
            <LinearProgress sx={{ flexGrow: 1 }} />
            <Typography>Loading documents...</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="between" alignItems="center" mb={3}>
          <Typography variant="h6">
            Documents ({total})
          </Typography>
          <Button
            startIcon={<RefreshIcon />}
            onClick={loadDocuments}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>

        {/* Filters */}
        <Box display="flex" gap={2} mb={3} flexWrap="wrap">
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>File Type</InputLabel>
            <Select
              value={filters.file_type}
              label="File Type"
              onChange={(e) => handleFilterChange('file_type', e.target.value)}
            >
              <MenuItem value="">All</MenuItem>
              <MenuItem value="pdf">PDF</MenuItem>
              <MenuItem value="docx">DOCX</MenuItem>
              <MenuItem value="txt">TXT</MenuItem>
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel>Status</InputLabel>
            <Select
              value={filters.processing_status}
              label="Status"
              onChange={(e) => handleFilterChange('processing_status', e.target.value)}
            >
              <MenuItem value="">All</MenuItem>
              <MenuItem value="pending">Pending</MenuItem>
              <MenuItem value="processing">Processing</MenuItem>
              <MenuItem value="completed">Completed</MenuItem>
              <MenuItem value="failed">Failed</MenuItem>
            </Select>
          </FormControl>

          <TextField
            size="small"
            label="Tags"
            placeholder="Filter by tags..."
            value={filters.tags}
            onChange={(e) => handleFilterChange('tags', e.target.value)}
            sx={{ minWidth: 200 }}
          />
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {documents.length === 0 ? (
          <Box textAlign="center" py={4}>
            <DocumentIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary">
              No documents found
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Upload a document to get started
            </Typography>
          </Box>
        ) : (
          <>
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Size</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Chunks</TableCell>
                    <TableCell>Tags</TableCell>
                    <TableCell>Upload Date</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {documents.map((document) => (
                    <TableRow
                      key={document.id}
                      hover
                      sx={{ cursor: 'pointer' }}
                      onClick={() => onDocumentSelect?.(document)}
                    >
                      <TableCell>
                        <Box display="flex" alignItems="center" gap={1}>
                          <DocumentIcon color="action" />
                          <Box>
                            <Typography variant="body2" fontWeight="medium">
                              {document.original_filename}
                            </Typography>
                            {document.content_preview && (
                              <Typography variant="caption" color="text.secondary">
                                {document.content_preview.substring(0, 60)}...
                              </Typography>
                            )}
                          </Box>
                        </Box>
                      </TableCell>
                      
                      <TableCell>
                        <Chip
                          label={document.file_type.toUpperCase()}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      
                      <TableCell>
                        {formatFileSize(document.file_size_mb)}
                      </TableCell>
                      
                      <TableCell>
                        <Box display="flex" alignItems="center" gap={1}>
                          {getStatusIcon(document.processing_status)}
                          <Chip
                            label={document.processing_status}
                            size="small"
                            color={getStatusColor(document.processing_status)}
                          />
                        </Box>
                        {document.processing_error && (
                          <Typography variant="caption" color="error">
                            {document.processing_error}
                          </Typography>
                        )}
                      </TableCell>
                      
                      <TableCell>
                        {document.total_chunks || '-'}
                      </TableCell>
                      
                      <TableCell>
                        <Box display="flex" flexWrap="wrap" gap={0.5}>
                          {document.tags?.slice(0, 2).map((tag, index) => (
                            <Chip key={index} label={tag} size="small" />
                          ))}
                          {document.tags && document.tags.length > 2 && (
                            <Chip label={`+${document.tags.length - 2}`} size="small" />
                          )}
                        </Box>
                      </TableCell>
                      
                      <TableCell>
                        <Typography variant="body2">
                          {formatDate(document.created_at)}
                        </Typography>
                      </TableCell>
                      
                      <TableCell align="right">
                        <Tooltip title="Delete document">
                          <IconButton
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              setDeleteDialog({ open: true, document });
                            }}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            {/* Pagination */}
            {totalPages > 1 && (
              <Box display="flex" justifyContent="center" mt={3}>
                <Pagination
                  count={totalPages}
                  page={page}
                  onChange={(_, newPage) => setPage(newPage)}
                  color="primary"
                />
              </Box>
            )}
          </>
        )}

        {/* Delete Confirmation Dialog */}
        <Dialog
          open={deleteDialog.open}
          onClose={() => setDeleteDialog({ open: false, document: null })}
        >
          <DialogTitle>Delete Document</DialogTitle>
          <DialogContent>
            <Typography>
              Are you sure you want to delete "{deleteDialog.document?.original_filename}"?
              This action cannot be undone.
            </Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDeleteDialog({ open: false, document: null })}>
              Cancel
            </Button>
            <Button
              onClick={() => deleteDialog.document && handleDeleteDocument(deleteDialog.document)}
              color="error"
            >
              Delete
            </Button>
          </DialogActions>
        </Dialog>
      </CardContent>
    </Card>
  );
};