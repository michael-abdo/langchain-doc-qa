import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Container,
  Box,
  Typography,
  AppBar,
  Toolbar,
  Tabs,
  Tab,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  Description as DocumentIcon,
  QuestionAnswer as QueryIcon,
} from '@mui/icons-material';
import { DocumentUpload } from './components/DocumentUpload';
import { QueryInterface } from './components/QueryInterface';
import { DocumentList } from './components/DocumentList';
import { type Document } from './services/api';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
});

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index, ...other }: TabPanelProps) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

function App() {
  const [currentTab, setCurrentTab] = useState(0);
  const [sessionId, setSessionId] = useState<string>('');
  const [refreshDocuments, setRefreshDocuments] = useState(0);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'info',
  });

  useEffect(() => {
    // Generate a session ID for this browser session
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
  }, []);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleUploadComplete = (document: Document) => {
    setNotification({
      open: true,
      message: `Document "${document.original_filename}" uploaded successfully!`,
      severity: 'success',
    });
    setRefreshDocuments(prev => prev + 1);
    
    // Switch to query tab if we're on documents tab
    if (currentTab === 0) {
      setCurrentTab(1);
    }
  };

  const handleUploadError = (error: string) => {
    setNotification({
      open: true,
      message: `Upload failed: ${error}`,
      severity: 'error',
    });
  };

  const handleDocumentSelect = (document: Document) => {
    // Switch to query tab when a document is selected
    setCurrentTab(1);
    setNotification({
      open: true,
      message: `Selected document: ${document.original_filename}`,
      severity: 'info',
    });
  };

  const handleCloseNotification = () => {
    setNotification(prev => ({ ...prev, open: false }));
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static" elevation={1}>
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              LangChain Document Q&A
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.8 }}>
              Session: {sessionId.substr(-8)}
            </Typography>
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          <Typography variant="h4" gutterBottom align="center">
            Document Question & Answer System
          </Typography>
          
          <Typography variant="body1" gutterBottom align="center" color="text.secondary" sx={{ mb: 4 }}>
            Upload documents and ask questions to get AI-powered answers with source citations
          </Typography>

          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
            <Tabs
              value={currentTab}
              onChange={handleTabChange}
              aria-label="document qa tabs"
              centered
            >
              <Tab
                icon={<DocumentIcon />}
                label="Documents"
                id="tab-0"
                aria-controls="tabpanel-0"
              />
              <Tab
                icon={<QueryIcon />}
                label="Ask Questions"
                id="tab-1"
                aria-controls="tabpanel-1"
              />
            </Tabs>
          </Box>

          <TabPanel value={currentTab} index={0}>
            <DocumentUpload
              onUploadComplete={handleUploadComplete}
              onUploadError={handleUploadError}
            />
            <DocumentList
              refreshTrigger={refreshDocuments}
              onDocumentSelect={handleDocumentSelect}
            />
          </TabPanel>

          <TabPanel value={currentTab} index={1}>
            <QueryInterface
              sessionId={sessionId}
              onSessionUpdate={setSessionId}
            />
          </TabPanel>
        </Container>

        <Snackbar
          open={notification.open}
          autoHideDuration={6000}
          onClose={handleCloseNotification}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert
            onClose={handleCloseNotification}
            severity={notification.severity}
            sx={{ width: '100%' }}
          >
            {notification.message}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}

export default App;
