"""
RAG Pipeline Orchestrator

Provides end-to-end RAG pipeline execution with comprehensive error handling,
graceful degradation, and detailed logging. Ties together Query Processing,
Retrieval, and Answer Generation services.
"""

import asyncio
import time
import traceback
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from contextlib import contextmanager

from app.core.common import (
    BaseService,
    get_service_logger,
    config,
    with_error_handling,
    datetime
)
from app.core.exceptions import (
    BaseAppException,
    QueryValidationError,
    SecurityError,
    RateLimitError,
    ComplexityError,
    VectorStoreError,
    LLMError,
    ProcessingError,
    ValidationError
)
from app.services.query_processor import QueryProcessor, ProcessedQuery
from app.services.retrieval_engine import RetrievalEngine, SearchResults
from app.services.answer_generator import AnswerGenerator, Answer


class ErrorCategory(Enum):
    """Categories of errors for handling and reporting."""
    USER_INPUT = "user_input"
    SECURITY = "security"
    RATE_LIMIT = "rate_limit"
    RESOURCE_LIMIT = "resource_limit"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


@dataclass
class PipelineError:
    """Structured error information."""
    category: ErrorCategory
    stage: str
    error_type: str
    message: str
    user_message: str
    details: Dict[str, Any]
    timestamp: datetime
    traceback: Optional[str] = None
    recoverable: bool = False


@dataclass
class PipelineContext:
    """Context information for pipeline execution."""
    pipeline_id: str
    user_id: str
    start_time: float
    stages_completed: List[str] = field(default_factory=list)
    errors: List[PipelineError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    partial_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    success: bool
    answer: Optional[Answer]
    query: Optional[ProcessedQuery]
    search_results: Optional[SearchResults]
    errors: List[PipelineError]
    execution_time: float
    pipeline_id: str
    degraded: bool = False
    degradation_reason: Optional[str] = None


class ErrorCategorizer:
    """Categorizes errors for appropriate handling."""
    
    @staticmethod
    def categorize(error: Exception, stage: str) -> ErrorCategory:
        """Categorize an error based on type and stage."""
        if isinstance(error, (QueryValidationError, ValidationError)):
            return ErrorCategory.USER_INPUT
        elif isinstance(error, SecurityError):
            return ErrorCategory.SECURITY
        elif isinstance(error, RateLimitError):
            return ErrorCategory.RATE_LIMIT
        elif isinstance(error, ComplexityError):
            return ErrorCategory.RESOURCE_LIMIT
        elif isinstance(error, VectorStoreError):
            return ErrorCategory.RETRIEVAL
        elif isinstance(error, LLMError):
            return ErrorCategory.GENERATION
        elif "timeout" in str(error).lower():
            return ErrorCategory.EXTERNAL_SERVICE
        elif isinstance(error, BaseAppException):
            return ErrorCategory.INTERNAL
        else:
            return ErrorCategory.UNKNOWN


class UserMessageGenerator:
    """Generates user-friendly error messages."""
    
    USER_MESSAGES = {
        ErrorCategory.USER_INPUT: {
            "default": "Your query needs adjustment. Please check and try again.",
            "too_short": "Your query is too short. Please provide more detail.",
            "too_long": "Your query is too long. Please make it more concise.",
            "invalid_format": "Your query format is not recognized. Please rephrase."
        },
        ErrorCategory.SECURITY: {
            "default": "Your query was blocked for security reasons.",
            "injection": "Potentially harmful content detected in your query.",
            "prompt_injection": "Your query appears to contain restricted patterns."
        },
        ErrorCategory.RATE_LIMIT: {
            "default": "You've made too many requests. Please wait a moment.",
            "specific": "Please wait {wait_time} seconds before trying again."
        },
        ErrorCategory.RESOURCE_LIMIT: {
            "default": "Your query is too complex to process.",
            "complexity": "Please simplify your query and try again."
        },
        ErrorCategory.RETRIEVAL: {
            "default": "Unable to search for relevant information.",
            "no_results": "No relevant information found for your query.",
            "timeout": "Search took too long. Please try a simpler query."
        },
        ErrorCategory.GENERATION: {
            "default": "Unable to generate a response.",
            "api_error": "The AI service is temporarily unavailable.",
            "timeout": "Response generation timed out. Please try again."
        },
        ErrorCategory.EXTERNAL_SERVICE: {
            "default": "An external service is temporarily unavailable.",
            "timeout": "The service is taking too long to respond."
        },
        ErrorCategory.INTERNAL: {
            "default": "An internal error occurred. Please try again later."
        },
        ErrorCategory.UNKNOWN: {
            "default": "An unexpected error occurred. Please try again."
        }
    }
    
    @classmethod
    def generate_message(cls, category: ErrorCategory, error: Exception, details: Dict[str, Any] = None) -> str:
        """Generate a user-friendly error message."""
        messages = cls.USER_MESSAGES.get(category, cls.USER_MESSAGES[ErrorCategory.UNKNOWN])
        
        # Try to find a specific message based on error details
        if details:
            if category == ErrorCategory.RATE_LIMIT and "retry_after" in details:
                return messages["specific"].format(wait_time=details["retry_after"])
            elif category == ErrorCategory.USER_INPUT:
                if "too short" in str(error).lower():
                    return messages.get("too_short", messages["default"])
                elif "too long" in str(error).lower():
                    return messages.get("too_long", messages["default"])
        
        # Check error message for keywords
        error_str = str(error).lower()
        for key, message in messages.items():
            if key != "default" and key in error_str:
                return message
        
        return messages["default"]


class GracefulDegradation:
    """Handles graceful degradation strategies."""
    
    @staticmethod
    def can_degrade(context: PipelineContext, error: PipelineError) -> bool:
        """Determine if graceful degradation is possible."""
        # Can degrade if we have partial results
        if context.partial_results:
            if error.category in [ErrorCategory.GENERATION, ErrorCategory.EXTERNAL_SERVICE]:
                return True
            if error.category == ErrorCategory.RETRIEVAL and "query" in context.partial_results:
                return True
        return False
    
    @staticmethod
    def apply_degradation(context: PipelineContext, error: PipelineError) -> Tuple[bool, Any, str]:
        """
        Apply degradation strategy.
        
        Returns:
            Tuple of (success, result, reason)
        """
        if error.category == ErrorCategory.GENERATION:
            # If generation failed but we have search results, return them
            if "search_results" in context.partial_results:
                search_results = context.partial_results["search_results"]
                if search_results.results:
                    # Create a simple answer from search results
                    content = "Based on the search results:\n\n"
                    for i, result in enumerate(search_results.results[:3]):
                        content += f"{i+1}. {result.document.page_content[:200]}...\n\n"
                    
                    # Create degraded answer
                    answer = Answer(
                        content=content,
                        confidence_score=0.5,  # Lower confidence for degraded response
                        sources=[r.source for r in search_results.results],
                        citations=[],
                        generation_time=0,
                        model_used="none",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_cost=0,
                        quality_metrics={"degraded": True},
                        safety_flags=[],
                        metadata={"degradation_reason": "generation_failed"}
                    )
                    return True, answer, "Generated summary from search results"
        
        elif error.category == ErrorCategory.RETRIEVAL:
            # If retrieval failed but we have a query, provide generic response
            if "query" in context.partial_results:
                query = context.partial_results["query"]
                answer = Answer(
                    content=f"I understand you're asking about '{query.sanitized_query}', but I'm unable to retrieve relevant information at this time. Please try rephrasing your question or try again later.",
                    confidence_score=0.1,
                    sources=[],
                    citations=[],
                    generation_time=0,
                    model_used="none",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_cost=0,
                    quality_metrics={"degraded": True},
                    safety_flags=[],
                    metadata={"degradation_reason": "retrieval_failed"}
                )
                return True, answer, "Provided generic response due to retrieval failure"
        
        return False, None, "No degradation strategy available"


class DetailedLogger:
    """Provides detailed logging with full context."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def log_pipeline_start(self, context: PipelineContext, query: str):
        """Log pipeline start with context."""
        self.logger.info(
            "rag_pipeline_started",
            pipeline_id=context.pipeline_id,
            user_id=context.user_id,
            query_preview=query[:100] + "..." if len(query) > 100 else query,
            timestamp=datetime.now().isoformat()
        )
    
    def log_stage_completion(self, context: PipelineContext, stage: str, result: Any):
        """Log successful stage completion."""
        self.logger.info(
            f"rag_pipeline_stage_completed",
            pipeline_id=context.pipeline_id,
            stage=stage,
            stages_completed=context.stages_completed,
            elapsed_time=time.time() - context.start_time,
            result_type=type(result).__name__
        )
    
    def log_error(self, context: PipelineContext, error: PipelineError):
        """Log error with full context."""
        self.logger.error(
            f"rag_pipeline_error",
            pipeline_id=context.pipeline_id,
            stage=error.stage,
            error_category=error.category.value,
            error_type=error.error_type,
            error_message=error.message,
            stages_completed=context.stages_completed,
            elapsed_time=time.time() - context.start_time,
            error_details=error.details,
            traceback=error.traceback
        )
    
    def log_degradation(self, context: PipelineContext, reason: str):
        """Log graceful degradation."""
        self.logger.warning(
            "rag_pipeline_degraded",
            pipeline_id=context.pipeline_id,
            reason=reason,
            stages_completed=context.stages_completed,
            elapsed_time=time.time() - context.start_time
        )
    
    def log_pipeline_complete(self, context: PipelineContext, result: PipelineResult):
        """Log pipeline completion."""
        self.logger.info(
            "rag_pipeline_completed",
            pipeline_id=context.pipeline_id,
            success=result.success,
            degraded=result.degraded,
            execution_time=result.execution_time,
            stages_completed=context.stages_completed,
            error_count=len(result.errors),
            error_categories=[e.category.value for e in result.errors]
        )


class RAGPipeline(BaseService):
    """Main RAG Pipeline orchestrator with comprehensive error handling."""
    
    def __init__(self):
        super().__init__("RAGPipeline")
        self.query_processor = QueryProcessor()
        self.retrieval_engine = RetrievalEngine()
        self.answer_generator = AnswerGenerator()
        self.error_categorizer = ErrorCategorizer()
        self.message_generator = UserMessageGenerator()
        self.degradation_handler = GracefulDegradation()
        self.detailed_logger = DetailedLogger(self.logger)
        self._initialize()
    
    @with_error_handling("initialization")
    def _initialize(self):
        """Initialize the RAG pipeline."""
        self.logger.info("Initializing RAG Pipeline")
        self._initialized = True
    
    def process_query(self, query: str, user_id: str) -> PipelineResult:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query: User's query
            user_id: User identifier
            
        Returns:
            PipelineResult with answer or error information
        """
        # Create pipeline context
        context = PipelineContext(
            pipeline_id=str(uuid.uuid4()),
            user_id=user_id,
            start_time=time.time()
        )
        
        # Log pipeline start
        self.detailed_logger.log_pipeline_start(context, query)
        
        try:
            # Stage 1: Query Processing
            processed_query = self._execute_stage(
                context, 
                "query_processing",
                lambda: self.query_processor.process_query(query, user_id)
            )
            if not processed_query:
                return self._create_error_result(context)
            
            context.partial_results["query"] = processed_query
            
            # Stage 2: Document Retrieval
            search_results = self._execute_stage(
                context,
                "retrieval",
                lambda: self.retrieval_engine.search_documents(processed_query)
            )
            if not search_results:
                # Try degradation
                return self._try_degradation_or_fail(context)
            
            context.partial_results["search_results"] = search_results
            
            # Stage 3: Answer Generation
            answer = self._execute_stage(
                context,
                "generation",
                lambda: self.answer_generator.generate_answer(processed_query, search_results)
            )
            if not answer:
                # Try degradation
                return self._try_degradation_or_fail(context)
            
            # Success - create result
            execution_time = time.time() - context.start_time
            result = PipelineResult(
                success=True,
                answer=answer,
                query=processed_query,
                search_results=search_results,
                errors=[],
                execution_time=execution_time,
                pipeline_id=context.pipeline_id,
                degraded=False
            )
            
            self.detailed_logger.log_pipeline_complete(context, result)
            return result
            
        except Exception as e:
            # Catch-all for unexpected errors
            error = self._create_pipeline_error(e, "pipeline", context)
            context.errors.append(error)
            return self._create_error_result(context)
    
    def _execute_stage(self, context: PipelineContext, stage: str, operation: callable) -> Optional[Any]:
        """
        Execute a pipeline stage with error handling.
        
        Returns:
            Result of operation or None if failed
        """
        try:
            result = operation()
            context.stages_completed.append(stage)
            self.detailed_logger.log_stage_completion(context, stage, result)
            return result
            
        except Exception as e:
            error = self._create_pipeline_error(e, stage, context)
            context.errors.append(error)
            self.detailed_logger.log_error(context, error)
            return None
    
    def _create_pipeline_error(self, exception: Exception, stage: str, context: PipelineContext) -> PipelineError:
        """Create a structured pipeline error."""
        category = self.error_categorizer.categorize(exception, stage)
        
        # Extract details from exception
        details = {}
        if hasattr(exception, '__dict__'):
            details = {k: v for k, v in exception.__dict__.items() if not k.startswith('_')}
        
        # Special handling for specific error types
        if isinstance(exception, RateLimitError) and hasattr(exception, 'retry_after'):
            details['retry_after'] = exception.retry_after
        
        user_message = self.message_generator.generate_message(category, exception, details)
        
        # Determine if error is recoverable
        recoverable = category in [
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorCategory.RETRIEVAL
        ]
        
        return PipelineError(
            category=category,
            stage=stage,
            error_type=type(exception).__name__,
            message=str(exception),
            user_message=user_message,
            details=details,
            timestamp=datetime.now(),
            traceback=traceback.format_exc(),  # Always include traceback for comprehensive error handling
            recoverable=recoverable
        )
    
    def _try_degradation_or_fail(self, context: PipelineContext) -> PipelineResult:
        """Try graceful degradation or create error result."""
        if context.errors:
            last_error = context.errors[-1]
            
            if self.degradation_handler.can_degrade(context, last_error):
                success, result, reason = self.degradation_handler.apply_degradation(
                    context, last_error
                )
                
                if success:
                    self.detailed_logger.log_degradation(context, reason)
                    
                    execution_time = time.time() - context.start_time
                    return PipelineResult(
                        success=True,
                        answer=result,
                        query=context.partial_results.get("query"),
                        search_results=context.partial_results.get("search_results"),
                        errors=context.errors,
                        execution_time=execution_time,
                        pipeline_id=context.pipeline_id,
                        degraded=True,
                        degradation_reason=reason
                    )
        
        return self._create_error_result(context)
    
    def _create_error_result(self, context: PipelineContext) -> PipelineResult:
        """Create an error result from context."""
        execution_time = time.time() - context.start_time
        
        result = PipelineResult(
            success=False,
            answer=None,
            query=context.partial_results.get("query"),
            search_results=context.partial_results.get("search_results"),
            errors=context.errors,
            execution_time=execution_time,
            pipeline_id=context.pipeline_id,
            degraded=False
        )
        
        self.detailed_logger.log_pipeline_complete(context, result)
        return result
    
    def get_error_summary(self, result: PipelineResult) -> Dict[str, Any]:
        """Get a summary of errors from pipeline result."""
        if not result.errors:
            return {"error_count": 0, "categories": []}
        
        categories = {}
        for error in result.errors:
            category = error.category.value
            if category not in categories:
                categories[category] = []
            categories[category].append({
                "stage": error.stage,
                "message": error.user_message,
                "recoverable": error.recoverable
            })
        
        return {
            "error_count": len(result.errors),
            "categories": categories,
            "first_error": result.errors[0].user_message if result.errors else None,
            "all_recoverable": all(e.recoverable for e in result.errors)
        }
    
    def shutdown(self):
        """Shutdown the pipeline and all services."""
        self.logger.info("Shutting down RAG Pipeline")
        self.retrieval_engine.shutdown()
        self.answer_generator.shutdown()
        self.logger.info("RAG Pipeline shutdown complete")