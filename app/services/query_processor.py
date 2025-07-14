"""
Query Processing Service for RAG Pipeline

Provides secure query ingestion, validation, and preprocessing with comprehensive
security features including malicious input detection, complexity analysis, and
rate limiting.
"""

import re
import time
import html
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import tiktoken
from app.core.common import (
    BaseService,
    get_service_logger,
    config,
    CommonValidators,
    with_error_handling,
    ensure_directory_exists,
    normalize_text
)
from app.core.exceptions import (
    QueryValidationError,
    SecurityError,
    RateLimitError,
    ComplexityError,
    ProcessingError
)


@dataclass
class ProcessedQuery:
    """Processed and validated query ready for retrieval."""
    original_query: str
    sanitized_query: str
    normalized_query: str
    complexity_score: float
    token_count: int
    detected_intent: str
    language: str
    timestamp: datetime
    user_id: str
    processing_time: float
    security_flags: List[str]
    metadata: Dict[str, Any]


@dataclass
class ValidationResult:
    """Result of query validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    security_issues: List[str]


@dataclass
class ComplexityScore:
    """Query complexity analysis result."""
    score: float  # 0-100 scale
    factors: Dict[str, float]
    estimated_processing_time: float
    resource_requirements: Dict[str, Any]


class InputSanitizer:
    """Handles input sanitization and malicious pattern detection."""
    
    # Patterns for detecting various injection attempts
    SQL_INJECTION_PATTERNS = [
        r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b.*\b(from|where|table|database)\b)",
        r"(--|\#|\/\*|\*\/|xp_|sp_)",
        r"(\b(or|and)\b\s*\d+\s*=\s*\d+)",
        r"('|\")\s*(or|and)\s*('|\")\s*('|\")",
        r"(\b(sleep|benchmark|waitfor)\b.*\()",
    ]
    
    SCRIPT_INJECTION_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript\s*:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"eval\s*\(",
        r"expression\s*\(",
    ]
    
    PROMPT_INJECTION_PATTERNS = [
        r"(ignore|forget|disregard).*previous.*instructions",
        r"(ignore|forget|disregard).*above.*instructions",
        r"(ignore|override).*system.*prompt",
        r"reveal.*system.*prompt",
        r"show.*original.*instructions",
        r"what.*your.*instructions",
        r"repeat.*everything.*above",
        r"(act|pretend|roleplay).*as.*different.*system",
        r"new.*instructions.*follow",
        r"###.*SYSTEM.*###",
        r"\[INST\]|\[\/INST\]",
        r"<\|.*system.*\|>",
    ]
    
    def __init__(self, logger):
        self.logger = logger
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self.sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS]
        self.script_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.SCRIPT_INJECTION_PATTERNS]
        self.prompt_patterns = [re.compile(p, re.IGNORECASE) for p in self.PROMPT_INJECTION_PATTERNS]
    
    def sanitize(self, query: str) -> Tuple[str, List[str]]:
        """
        Sanitize input and detect security issues.
        
        Returns:
            Tuple of (sanitized_query, security_flags)
        """
        if not query:
            return "", []
        
        security_flags = []
        
        # HTML entity encoding
        sanitized = html.escape(query)
        
        # Check for SQL injection
        for pattern in self.sql_patterns:
            if pattern.search(query):
                security_flags.append("potential_sql_injection")
                self.logger.warning(f"Potential SQL injection detected: {pattern.pattern}")
                break
        
        # Check for script injection
        for pattern in self.script_patterns:
            if pattern.search(query):
                security_flags.append("potential_script_injection")
                self.logger.warning(f"Potential script injection detected: {pattern.pattern}")
                # Remove script tags and dangerous content
                sanitized = re.sub(r"<script[^>]*>.*?</script>", "", sanitized, flags=re.IGNORECASE | re.DOTALL)
                sanitized = re.sub(r"javascript\s*:", "", sanitized, flags=re.IGNORECASE)
                break
        
        # Check for prompt injection
        for pattern in self.prompt_patterns:
            if pattern.search(query):
                security_flags.append("potential_prompt_injection")
                self.logger.warning(f"Potential prompt injection detected: {pattern.pattern}")
                break
        
        # Remove any remaining HTML tags
        sanitized = re.sub(r"<[^>]+>", "", sanitized)
        
        # Remove excessive whitespace
        sanitized = " ".join(sanitized.split())
        
        # Decode HTML entities for readability
        sanitized = html.unescape(sanitized)
        
        return sanitized, security_flags


class QueryValidator:
    """Validates query structure and content."""
    
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 2000
    VALID_CHARS_PATTERN = re.compile(r"^[\w\s\-.,!?'\"():;@#$%&*+=\[\]{}|\\/<>~`]+$")
    
    def __init__(self, logger):
        self.logger = logger
    
    def validate(self, query: str) -> ValidationResult:
        """Validate query structure and content."""
        errors = []
        warnings = []
        security_issues = []
        
        # Length validation
        if len(query) < self.MIN_QUERY_LENGTH:
            errors.append(f"Query too short (minimum {self.MIN_QUERY_LENGTH} characters)")
        elif len(query) > self.MAX_QUERY_LENGTH:
            errors.append(f"Query too long (maximum {self.MAX_QUERY_LENGTH} characters)")
        
        # Character validation
        if not self.VALID_CHARS_PATTERN.match(query):
            invalid_chars = set(re.sub(r"[\w\s\-.,!?'\"():;@#$%&*+=\[\]{}|\\/<>~`]", "", query))
            warnings.append(f"Query contains potentially problematic characters: {invalid_chars}")
        
        # Check for suspicious patterns
        if query.count("'") > 10 or query.count('"') > 10:
            warnings.append("Excessive use of quotes detected")
        
        if re.search(r"(.)\1{10,}", query):
            warnings.append("Repeated character sequences detected")
        
        # Check for potential encoding issues
        try:
            query.encode('utf-8').decode('utf-8')
        except UnicodeError:
            errors.append("Query contains invalid Unicode characters")
        
        # Check for binary content
        if any(ord(char) < 32 and char not in '\t\n\r' for char in query):
            security_issues.append("Query contains control characters")
        
        is_valid = len(errors) == 0 and len(security_issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            security_issues=security_issues
        )


class ComplexityAnalyzer:
    """Analyzes query complexity for resource protection."""
    
    def __init__(self, logger):
        self.logger = logger
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def analyze(self, query: str) -> ComplexityScore:
        """Analyze query complexity and resource requirements."""
        factors = {}
        
        # Token count
        tokens = self.tokenizer.encode(query)
        token_count = len(tokens)
        factors['token_count'] = min(token_count / 500, 1.0)  # Normalize to 0-1
        
        # Unique words
        words = query.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        factors['unique_words'] = 1 - unique_ratio  # Higher score for repetitive queries
        
        # Special patterns that increase complexity
        pattern_score = 0
        
        # Technical jargon or code
        if re.search(r"(function|class|import|def|var|const|let)", query, re.IGNORECASE):
            pattern_score += 0.2
        
        # Mathematical expressions
        if re.search(r"[\d\+\-\*/=<>]+", query):
            pattern_score += 0.1
        
        # Multiple questions
        question_count = query.count("?")
        if question_count > 1:
            pattern_score += 0.1 * min(question_count, 3)
        
        # Nested structures (parentheses, brackets)
        nesting_score = (query.count("(") + query.count("[") + query.count("{")) / 10
        pattern_score += min(nesting_score, 0.3)
        
        factors['pattern_complexity'] = min(pattern_score, 1.0)
        
        # Query length factor
        length_factor = min(len(query) / 1000, 1.0)
        factors['length'] = length_factor
        
        # Calculate overall complexity score (0-100)
        weights = {
            'token_count': 0.3,
            'unique_words': 0.2,
            'pattern_complexity': 0.3,
            'length': 0.2
        }
        
        score = sum(factors.get(k, 0) * v for k, v in weights.items()) * 100
        
        # Estimate processing time (in seconds)
        base_time = 0.5
        token_time = token_count * 0.01
        complexity_time = score * 0.05
        estimated_time = base_time + token_time + complexity_time
        
        # Resource requirements
        resources = {
            'memory_mb': int(50 + score * 2),
            'cpu_cores': 1 if score < 50 else 2,
            'gpu_required': score > 80,
            'cache_recommended': score < 30
        }
        
        return ComplexityScore(
            score=round(score, 2),
            factors=factors,
            estimated_processing_time=round(estimated_time, 2),
            resource_requirements=resources
        )


class RateLimiter:
    """Implements rate limiting for query processing."""
    
    def __init__(self, logger, requests_per_minute: int = 10):
        self.logger = logger
        self.requests_per_minute = requests_per_minute
        self.user_requests = defaultdict(list)
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def check_rate_limit(self, user_id: str) -> Tuple[bool, Optional[int]]:
        """
        Check if user is within rate limit.
        
        Returns:
            Tuple of (is_allowed, seconds_until_reset)
        """
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries()
        
        # Get user's recent requests
        user_history = self.user_requests[user_id]
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        user_history = [t for t in user_history if t > cutoff_time]
        self.user_requests[user_id] = user_history
        
        # Check if within limit
        if len(user_history) >= self.requests_per_minute:
            oldest_request = min(user_history)
            seconds_until_reset = int(60 - (current_time - oldest_request))
            return False, seconds_until_reset
        
        # Add current request
        user_history.append(current_time)
        return True, None
    
    def _cleanup_old_entries(self):
        """Remove old entries from all users."""
        current_time = time.time()
        cutoff_time = current_time - 60
        
        for user_id in list(self.user_requests.keys()):
            self.user_requests[user_id] = [
                t for t in self.user_requests[user_id] if t > cutoff_time
            ]
            if not self.user_requests[user_id]:
                del self.user_requests[user_id]
        
        self.last_cleanup = current_time


class QueryProcessor(BaseService):
    """Main query processing service with security features."""
    
    def __init__(self):
        super().__init__("QueryProcessor")
        self.sanitizer = InputSanitizer(self.logger)
        self.validator = QueryValidator(self.logger)
        self.complexity_analyzer = ComplexityAnalyzer(self.logger)
        self.rate_limiter = RateLimiter(self.logger)
        self._initialize()
    
    @with_error_handling("initialization")
    def _initialize(self):
        """Initialize the query processor."""
        self.logger.info("Initializing Query Processor")
        self._initialized = True
    
    @with_error_handling("query_processing", raise_as=ProcessingError, reraise_if=(RateLimitError, SecurityError, QueryValidationError, ComplexityError))
    def process_query(self, query: str, user_id: str) -> ProcessedQuery:
        """
        Process and validate a user query.
        
        Args:
            query: Raw user query
            user_id: User identifier for rate limiting
            
        Returns:
            ProcessedQuery object ready for retrieval
            
        Raises:
            RateLimitError: If user exceeds rate limit
            SecurityError: If security issues detected
            QueryValidationError: If query validation fails
            ComplexityError: If query too complex
        """
        start_time = time.time()
        
        # Rate limiting
        is_allowed, wait_time = self.rate_limiter.check_rate_limit(user_id)
        if not is_allowed:
            raise RateLimitError(
                f"Rate limit exceeded. Please wait {wait_time} seconds.",
                retry_after=wait_time
            )
        
        # Input sanitization
        sanitized_query, security_flags = self.sanitizer.sanitize(query)
        
        # Block if critical security issues detected
        if any(flag in security_flags for flag in [
            "potential_sql_injection",
            "potential_script_injection",
            "potential_prompt_injection"
        ]):
            raise SecurityError(
                "Query contains potentially malicious content",
                security_flags=security_flags
            )
        
        # Query validation
        validation_result = self.validator.validate(sanitized_query)
        if not validation_result.is_valid:
            raise QueryValidationError(
                "Query validation failed",
                errors=validation_result.errors,
                warnings=validation_result.warnings
            )
        
        # Complexity analysis
        complexity_score = self.complexity_analyzer.analyze(sanitized_query)
        
        # Block if too complex
        if complexity_score.score > 80:
            raise ComplexityError(
                f"Query too complex (score: {complexity_score.score})",
                complexity_score=complexity_score
            )
        
        # Normalize query for better retrieval
        normalized_query = normalize_text(sanitized_query)
        
        # Detect query intent
        detected_intent = self._detect_intent(normalized_query)
        
        # Detect language
        language = self._detect_language(normalized_query)
        
        processing_time = time.time() - start_time
        
        return ProcessedQuery(
            original_query=query,
            sanitized_query=sanitized_query,
            normalized_query=normalized_query,
            complexity_score=complexity_score.score,
            token_count=len(self.complexity_analyzer.tokenizer.encode(sanitized_query)),
            detected_intent=detected_intent,
            language=language,
            timestamp=datetime.now(),
            user_id=user_id,
            processing_time=processing_time,
            security_flags=security_flags,
            metadata={
                'validation_warnings': validation_result.warnings,
                'complexity_factors': complexity_score.factors,
                'estimated_processing_time': complexity_score.estimated_processing_time,
                'resource_requirements': complexity_score.resource_requirements
            }
        )
    
    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query."""
        query_lower = query.lower()
        
        # Question patterns
        if any(query_lower.startswith(q) for q in ['what', 'who', 'where', 'when', 'why', 'how']):
            return "question"
        
        # Command patterns
        if any(word in query_lower for word in ['show', 'list', 'find', 'search', 'get']):
            return "search"
        
        # Explanation patterns
        if any(word in query_lower for word in ['explain', 'describe', 'define', 'tell me about']):
            return "explanation"
        
        # Comparison patterns
        if any(word in query_lower for word in ['compare', 'difference between', 'versus', 'vs']):
            return "comparison"
        
        # Analysis patterns
        if any(word in query_lower for word in ['analyze', 'evaluate', 'assess', 'review']):
            return "analysis"
        
        return "general"
    
    def _detect_language(self, query: str) -> str:
        """Simple language detection (can be enhanced with ML models)."""
        # For now, just check for common non-English characters
        # In production, use langdetect or similar library
        
        # Check for common non-English characters
        if re.search(r"[àáâãäåèéêëìíîïòóôõöùúûü]", query, re.IGNORECASE):
            return "non-english"
        
        # Default to English
        return "english"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query processing statistics."""
        return {
            'rate_limiter_users': len(self.rate_limiter.user_requests),
            'requests_per_minute_limit': self.rate_limiter.requests_per_minute,
            'max_query_length': QueryValidator.MAX_QUERY_LENGTH,
            'min_query_length': QueryValidator.MIN_QUERY_LENGTH
        }