# LangChain Document Q&A System

A minimal, production-ready document question-answering system built with LangChain that scales from POC to enterprise deployment.

## 🎯 Core Functionality

**INPUT:** Documents (PDF, DOCX, TXT files)  
**OUTPUT:** Answers to questions about those documents

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/langchain-doc-qa.git
cd langchain-doc-qa

# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the application
python -m app.api.main
```

## 📦 Architecture

```
langchain-doc-qa/
├── app/
│   ├── core/           # Configuration, models, database
│   ├── services/       # Document processing, embeddings, RAG
│   ├── api/            # FastAPI endpoints
│   └── frontend/       # Web interface
├── tests/              # Test suite
├── docker/             # Docker configuration
└── docs/               # Documentation
```

## 🔧 Features

- **Document Processing**: PDF, DOCX, TXT support with intelligent chunking
- **Vector Storage**: Efficient similarity search with future PostgreSQL+pgvector support
- **RAG Pipeline**: Context-aware question answering with source citations
- **REST API**: Production-ready FastAPI with streaming support
- **AWS Deployment**: Containerized deployment with ECS/Fargate

## 🔄 Advanced DRY Architecture

**Centralized Logic - Single Source of Truth:**

### **Configuration & Validation (`app/core/config.py`)**
- `validate_llm_health()` - LLM provider validation
- `validate_database_health()` - Database connectivity checks  
- `validate_vector_store_health()` - Vector store validation
- `validate_critical_startup_config()` - Startup validation

### **Error Response Factory (`app/core/exceptions.py`)**
- `create_error_response()` - Centralized JSON error response formatter
- `create_app_exception_response()` - Application exception responses
- `create_validation_error_response()` - Request validation responses
- `create_http_error_response()` - HTTP error responses
- `create_unexpected_error_response()` - Unexpected error responses
- `get_correlation_id_from_request()` - Request context utility

### **Logging & Utilities (`app/core/logging.py`)**
- `get_utc_timestamp()` - Standardized timestamp generation
- `get_utc_datetime()` - Datetime utility
- `generate_uuid()` - UUID generation  
- `log_request_error()` - Centralized request error logging

### **API Layer (`app/api/main.py`)**
- **Before:** 4 exception handlers with duplicate JSON response patterns
- **After:** All handlers use centralized response factories
- **Result:** 100% consistent error format across all endpoints

**Benefits:**
- ✅ **Zero Duplication**: All JSON responses use single factory
- ✅ **Consistent Format**: `{error: {code, message, details, correlation_id}}`
- ✅ **Single-Point Changes**: Update error format once, applies everywhere
- ✅ **Request Tracing**: Correlation IDs in all error responses
- ✅ **Code Reduction**: 80+ lines of duplicate code eliminated
- ✅ **Testing Simplified**: Mock single response factory vs. multiple handlers

## 🚧 Scaling Path

This POC is designed to scale to the full enterprise stack:

- **POC → Production**
  - FastAPI → LangServe
  - In-memory → PostgreSQL+pgvector
  - Simple RAG → LangGraph multi-agent
  - Basic UI → Streamlit dashboard
  - Minimal metrics → Apache Superset analytics

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Development Guide](docs/development.md)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.