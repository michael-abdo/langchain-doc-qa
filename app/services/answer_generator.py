"""
Answer Generation Service for RAG Pipeline

Provides high-quality response generation with LLM integration, including
prompt building, response validation, content filtering, and quality control.
"""

import asyncio
import time
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from app.core.common import (
    BaseService,
    get_service_logger,
    config,
    with_error_handling,
    normalize_text
)
from app.core.config import ConfigAccessor
from app.core.exceptions import (
    LLMError,
    ValidationError,
    ProcessingError,
    ExternalServiceError
)
from app.services.query_processor import ProcessedQuery
from app.services.retrieval_engine import SearchResults, SearchResult


@dataclass
class Answer:
    """Generated answer with metadata."""
    content: str
    confidence_score: float
    sources: List[str]
    citations: List[Dict[str, Any]]
    generation_time: float
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_cost: float
    quality_metrics: Dict[str, float]
    safety_flags: List[str]
    metadata: Dict[str, Any]


@dataclass
class Prompt:
    """Structured prompt for LLM."""
    system_message: str
    user_message: str
    context: str
    constraints: List[str]
    expected_format: str
    token_count: int


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    quality_score: float
    issues: List[str]
    suggestions: List[str]


@dataclass
class FilteredResponse:
    """Response after content filtering."""
    content: str
    was_filtered: bool
    filtered_sections: List[str]
    safety_score: float


class PromptBuilder:
    """Builds context-aware prompts for LLM."""
    
    SYSTEM_TEMPLATE = """You are a helpful AI assistant that provides accurate, informative answers based on the provided context. 

Your responses should be:
1. Accurate and factual, based only on the provided context
2. Clear and well-structured
3. Appropriately detailed for the question asked
4. Include citations when referencing specific information

If the context doesn't contain enough information to fully answer the question, acknowledge this limitation.
"""
    
    USER_TEMPLATE = """Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Please provide a comprehensive answer, citing relevant sections from the context when appropriate."""
    
    def __init__(self, logger):
        self.logger = logger
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def build_prompt(self, query: ProcessedQuery, results: SearchResults) -> Prompt:
        """Build a structured prompt from query and search results."""
        # Extract context from search results
        context_parts = []
        sources = []
        
        for i, result in enumerate(results.results):
            context_parts.append(f"[Source {i+1}: {result.source}]\n{result.document.page_content}\n")
            sources.append(result.source)
        
        context = "\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Build messages
        system_message = self.SYSTEM_TEMPLATE
        user_message = self.USER_TEMPLATE.format(
            context=context,
            question=query.sanitized_query
        )
        
        # Add constraints based on query type
        constraints = self._get_constraints(query)
        
        # Define expected format
        expected_format = self._get_expected_format(query)
        
        # Count tokens
        full_prompt = f"{system_message}\n\n{user_message}"
        token_count = len(self.tokenizer.encode(full_prompt))
        
        # Truncate context if too long
        max_context_tokens = 3000
        if token_count > max_context_tokens:
            self.logger.warning(f"Prompt too long ({token_count} tokens), truncating context")
            context = self._truncate_context(context, max_context_tokens - 500)
            user_message = self.USER_TEMPLATE.format(
                context=context,
                question=query.sanitized_query
            )
            token_count = len(self.tokenizer.encode(f"{system_message}\n\n{user_message}"))
        
        return Prompt(
            system_message=system_message,
            user_message=user_message,
            context=context,
            constraints=constraints,
            expected_format=expected_format,
            token_count=token_count
        )
    
    def _get_constraints(self, query: ProcessedQuery) -> List[str]:
        """Get constraints based on query type."""
        constraints = [
            "Base your answer only on the provided context",
            "Include citations for specific claims",
            "Be concise but comprehensive"
        ]
        
        if query.detected_intent == "question":
            constraints.append("Provide a direct answer to the question")
        elif query.detected_intent == "explanation":
            constraints.append("Provide a detailed explanation with examples")
        elif query.detected_intent == "comparison":
            constraints.append("Structure your answer as a clear comparison")
        elif query.detected_intent == "analysis":
            constraints.append("Provide a thorough analysis with key insights")
        
        return constraints
    
    def _get_expected_format(self, query: ProcessedQuery) -> str:
        """Get expected response format based on query."""
        if query.detected_intent == "comparison":
            return "A structured comparison with clear sections for each item"
        elif query.detected_intent == "analysis":
            return "An analytical response with introduction, main points, and conclusion"
        else:
            return "A clear, well-structured response with appropriate paragraphs"
    
    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """Truncate context to fit within token limit."""
        tokens = self.tokenizer.encode(context)
        if len(tokens) <= max_tokens:
            return context
        
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        return truncated_text + "\n\n[Context truncated due to length]"


class ResponseValidator:
    """Validates LLM responses for quality and safety."""
    
    MIN_RESPONSE_LENGTH = 50
    MAX_RESPONSE_LENGTH = 4000
    
    def __init__(self, logger):
        self.logger = logger
    
    def validate_response(self, response: str, query: ProcessedQuery, context: SearchResults) -> ValidationResult:
        """Validate response quality and appropriateness."""
        issues = []
        suggestions = []
        quality_scores = {}
        
        # Length validation
        if len(response) < self.MIN_RESPONSE_LENGTH:
            issues.append(f"Response too short ({len(response)} chars)")
            suggestions.append("Provide more detail in the response")
            quality_scores['length'] = 0.3
        elif len(response) > self.MAX_RESPONSE_LENGTH:
            issues.append(f"Response too long ({len(response)} chars)")
            suggestions.append("Consider being more concise")
            quality_scores['length'] = 0.7
        else:
            quality_scores['length'] = 1.0
        
        # Relevance check
        relevance_score = self._check_relevance(response, query.sanitized_query)
        quality_scores['relevance'] = relevance_score
        if relevance_score < 0.5:
            issues.append("Response may not adequately address the question")
            suggestions.append("Ensure the response directly answers the query")
        
        # Citation check (if context provided)
        if context.results:
            has_citations = self._check_citations(response)
            quality_scores['citations'] = 1.0 if has_citations else 0.5
            if not has_citations:
                suggestions.append("Consider adding citations to support claims")
        else:
            quality_scores['citations'] = 1.0  # No penalty if no context
        
        # Coherence check
        coherence_score = self._check_coherence(response)
        quality_scores['coherence'] = coherence_score
        if coherence_score < 0.7:
            issues.append("Response lacks coherence or structure")
            suggestions.append("Improve response organization and flow")
        
        # Factual consistency (basic check)
        consistency_score = self._check_factual_consistency(response, context)
        quality_scores['consistency'] = consistency_score
        if consistency_score < 0.8:
            issues.append("Response may contain information not supported by context")
            suggestions.append("Ensure all claims are supported by the provided context")
        
        # Calculate overall quality score
        overall_score = sum(quality_scores.values()) / len(quality_scores)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            quality_score=overall_score,
            issues=issues,
            suggestions=suggestions
        )
    
    def _check_relevance(self, response: str, query: str) -> float:
        """Check if response is relevant to the query."""
        # Simple keyword overlap check
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        query_words -= common_words
        response_words -= common_words
        
        if not query_words:
            return 1.0
        
        overlap = len(query_words.intersection(response_words))
        return min(overlap / len(query_words), 1.0)
    
    def _check_citations(self, response: str) -> bool:
        """Check if response includes citations."""
        citation_patterns = [
            r'\[Source \d+\]',
            r'\(Source \d+\)',
            r'According to Source \d+',
            r'As mentioned in Source \d+',
            r'\[\d+\]'
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, response):
                return True
        return False
    
    def _check_coherence(self, response: str) -> float:
        """Check response coherence and structure."""
        # Basic coherence checks
        sentences = response.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Check for proper sentence structure
        well_formed = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) > 3:
                well_formed += 1
        
        coherence = well_formed / max(len(sentences), 1)
        
        # Check for paragraph structure
        paragraphs = response.split('\n\n')
        if len(paragraphs) > 1:
            coherence = min(coherence + 0.2, 1.0)
        
        return coherence
    
    def _check_factual_consistency(self, response: str, context: SearchResults) -> float:
        """Check if response is consistent with provided context."""
        if not context.results:
            return 1.0  # No context to check against
        
        # Extract all context content
        context_text = " ".join([r.document.page_content for r in context.results])
        
        # This is a simplified check - in production, use more sophisticated methods
        # Check if response contains information not in context
        response_sentences = response.split('.')
        suspicious_sentences = 0
        
        for sentence in response_sentences:
            sentence = sentence.strip().lower()
            if len(sentence) > 20:  # Only check substantial sentences
                # Check if key terms from sentence appear in context
                key_terms = [word for word in sentence.split() if len(word) > 4]
                if key_terms:
                    found_terms = sum(1 for term in key_terms if term in context_text.lower())
                    if found_terms < len(key_terms) * 0.3:  # Less than 30% of terms found
                        suspicious_sentences += 1
        
        consistency = 1.0 - (suspicious_sentences / max(len(response_sentences), 1))
        return max(consistency, 0.5)  # Minimum score of 0.5


class ContentFilter:
    """Filters inappropriate or harmful content."""
    
    # Patterns for potentially harmful content
    HARMFUL_PATTERNS = [
        # Violence
        r'\b(kill|murder|torture|harm|hurt|attack|assault)\b',
        # Hate speech
        r'\b(hate|discriminate|racist|sexist)\b',
        # Personal information
        r'\b(SSN|social security|credit card|password)\b',
        # Medical advice
        r'\b(diagnose|prescribe|medical advice|treatment for)\b',
        # Legal advice
        r'\b(legal advice|sue|lawsuit|attorney)\b',
    ]
    
    def __init__(self, logger):
        self.logger = logger
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.HARMFUL_PATTERNS]
    
    def filter_content(self, response: str) -> FilteredResponse:
        """Filter potentially harmful content from response."""
        filtered_sections = []
        safety_score = 1.0
        filtered_response = response
        
        for pattern in self._compiled_patterns:
            matches = pattern.findall(response)
            if matches:
                safety_score -= 0.1 * len(matches)
                for match in matches:
                    filtered_sections.append(f"Removed potentially harmful content: {match}")
                    # In production, implement more sophisticated filtering
                    self.logger.warning(f"Potentially harmful content detected: {match}")
        
        # Check for all caps (shouting)
        caps_ratio = sum(1 for c in response if c.isupper()) / max(len(response), 1)
        if caps_ratio > 0.3:
            safety_score -= 0.2
            filtered_sections.append("Excessive capitalization detected")
        
        # Ensure safety score is between 0 and 1
        safety_score = max(0, min(1, safety_score))
        
        # Apply filtering if safety score is too low
        if safety_score < 0.7:
            filtered_response = self._apply_filtering(response)
            was_filtered = True
        else:
            was_filtered = False
        
        return FilteredResponse(
            content=filtered_response,
            was_filtered=was_filtered,
            filtered_sections=filtered_sections,
            safety_score=safety_score
        )
    
    def _apply_filtering(self, response: str) -> str:
        """Apply content filtering to response."""
        # This is a placeholder - in production, implement sophisticated filtering
        filtered = response
        
        # Remove potentially harmful content
        for pattern in self._compiled_patterns:
            filtered = pattern.sub("[FILTERED]", filtered)
        
        return filtered


class LLMValidator:
    """Validates LLM API responses and handles failures."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def validate_api_response(self, response: Any, provider: str) -> Tuple[bool, Optional[str]]:
        """Validate LLM API response structure."""
        try:
            if provider == "openai":
                # Check OpenAI response structure
                if not hasattr(response, 'choices') or not response.choices:
                    return False, "Invalid OpenAI response: no choices"
                
                choice = response.choices[0]
                if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
                    return False, "Invalid OpenAI response: no message content"
                
                if not choice.message.content:
                    return False, "Empty response from OpenAI"
                
                return True, None
                
            elif provider == "anthropic":
                # Check Anthropic response structure
                if not hasattr(response, 'content') or not response.content:
                    return False, "Invalid Anthropic response: no content"
                
                if isinstance(response.content, list) and len(response.content) > 0:
                    if not hasattr(response.content[0], 'text'):
                        return False, "Invalid Anthropic response: no text in content"
                    
                    if not response.content[0].text:
                        return False, "Empty response from Anthropic"
                else:
                    return False, "Invalid Anthropic response structure"
                
                return True, None
                
            else:
                return False, f"Unknown provider: {provider}"
                
        except Exception as e:
            return False, f"Error validating response: {str(e)}"
    
    def detect_api_errors(self, response: Any, provider: str) -> Optional[str]:
        """Detect common API errors in response."""
        try:
            if hasattr(response, 'error'):
                return f"API error: {response.error}"
            
            # Check for rate limiting
            if hasattr(response, 'headers'):
                if 'x-ratelimit-remaining' in response.headers:
                    remaining = int(response.headers['x-ratelimit-remaining'])
                    if remaining == 0:
                        return "Rate limit exceeded"
            
            # Check for timeout indicators
            if hasattr(response, 'status_code'):
                if response.status_code == 408:
                    return "Request timeout"
                elif response.status_code == 503:
                    return "Service unavailable"
                elif response.status_code >= 500:
                    return f"Server error: {response.status_code}"
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting API errors: {str(e)}")
            return None


class AnswerGenerator(BaseService):
    """Main answer generation service with LLM integration."""
    
    RESPONSE_TIMEOUT = 60  # seconds
    MAX_RETRIES = 3
    
    def __init__(self):
        super().__init__("AnswerGenerator")
        self.config_accessor = config  # config is already a ConfigAccessor instance
        self.prompt_builder = PromptBuilder(self.logger)
        self.response_validator = ResponseValidator(self.logger)
        self.content_filter = ContentFilter(self.logger)
        self.llm_validator = LLMValidator(self.logger)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._llm_cache = {}
        self._initialize()
    
    @with_error_handling("initialization")
    def _initialize(self):
        """Initialize the answer generator."""
        self.logger.info("Initializing Answer Generator")
        
        # Initialize LLM based on provider
        llm_config = self.config_accessor.llm_config
        
        if llm_config['provider'] == 'openai':
            self._llm = ChatOpenAI(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens'],
                api_key=llm_config['api_key']
            )
        elif llm_config['provider'] == 'anthropic':
            self._llm = ChatAnthropic(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens'],
                api_key=llm_config['api_key']
            )
        else:
            raise LLMError(f"Unsupported LLM provider: {llm_config['provider']}")
        
        self._initialized = True
    
    @with_error_handling("answer_generation", raise_as=LLMError)
    def generate_answer(self, query: ProcessedQuery, context: SearchResults) -> Answer:
        """
        Generate answer using LLM with all safety features.
        
        Args:
            query: Processed query from QueryProcessor
            context: Search results from RetrievalEngine
            
        Returns:
            Answer object with content and metadata
        """
        start_time = time.time()
        
        # Build prompt
        prompt = self.prompt_builder.build_prompt(query, context)
        
        # Generate response with timeout
        try:
            future = self._executor.submit(self._generate_with_llm, prompt)
            response_data = future.result(timeout=self.RESPONSE_TIMEOUT)
        except FutureTimeoutError:
            raise LLMError(
                f"Response generation timeout exceeded ({self.RESPONSE_TIMEOUT}s)",
                provider=self.config_accessor.llm_config['provider']
            )
        
        # Extract response content
        response_content = response_data['content']
        
        # Validate response
        validation_result = self.response_validator.validate_response(
            response_content, query, context
        )
        
        # Filter content
        filtered_response = self.content_filter.filter_content(response_content)
        
        # Extract sources and citations
        sources = [r.source for r in context.results]
        citations = self._extract_citations(filtered_response.content, context)
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Calculate costs (simplified - adjust based on actual pricing)
        prompt_tokens = prompt.token_count
        completion_tokens = response_data.get('completion_tokens', 0)
        total_cost = self._calculate_cost(prompt_tokens, completion_tokens)
        
        # Compile quality metrics
        quality_metrics = {
            'validation_score': validation_result.quality_score,
            'safety_score': filtered_response.safety_score,
            'relevance_score': validation_result.quality_score * 0.8 + filtered_response.safety_score * 0.2
        }
        
        # Determine confidence score
        confidence_score = self._calculate_confidence(
            validation_result, filtered_response, context
        )
        
        return Answer(
            content=filtered_response.content,
            confidence_score=confidence_score,
            sources=sources,
            citations=citations,
            generation_time=generation_time,
            model_used=self.config_accessor.llm_config['model'],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost=total_cost,
            quality_metrics=quality_metrics,
            safety_flags=filtered_response.filtered_sections,
            metadata={
                'query_complexity': query.complexity_score,
                'context_documents': len(context.results),
                'was_filtered': filtered_response.was_filtered,
                'validation_issues': validation_result.issues,
                'suggestions': validation_result.suggestions
            }
        )
    
    def _generate_with_llm(self, prompt: Prompt) -> Dict[str, Any]:
        """Generate response using LLM with retries."""
        messages = [
            SystemMessage(content=prompt.system_message),
            HumanMessage(content=prompt.user_message)
        ]
        
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                # Make LLM call
                response = self._llm.invoke(messages)
                
                # Validate response
                is_valid, error = self.llm_validator.validate_api_response(
                    response, 
                    self.config_accessor.llm_config['provider']
                )
                
                if not is_valid:
                    raise LLMError(f"Invalid LLM response: {error}")
                
                # Extract content based on provider
                if self.config_accessor.llm_config['provider'] == 'openai':
                    content = response.content
                    completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0
                else:  # anthropic
                    content = response.content
                    completion_tokens = len(content.split()) * 1.3  # Rough estimate
                
                return {
                    'content': content,
                    'completion_tokens': int(completion_tokens)
                }
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"LLM generation attempt {attempt + 1} failed: {str(e)}"
                )
                
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        raise LLMError(
            f"Failed to generate response after {self.MAX_RETRIES} attempts: {str(last_error)}",
            provider=self.config_accessor.llm_config['provider']
        )
    
    def _extract_citations(self, response: str, context: SearchResults) -> List[Dict[str, Any]]:
        """Extract citations from response."""
        citations = []
        
        # Find all citation patterns
        citation_pattern = r'\[Source (\d+)\]'
        matches = re.finditer(citation_pattern, response)
        
        for match in matches:
            source_idx = int(match.group(1)) - 1  # Convert to 0-based index
            if 0 <= source_idx < len(context.results):
                result = context.results[source_idx]
                citations.append({
                    'source': result.source,
                    'chunk_id': result.chunk_id,
                    'relevance_score': result.relevance_score,
                    'position': match.start()
                })
        
        return citations
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost of LLM usage."""
        # Simplified cost calculation - adjust based on actual pricing
        model = self.config_accessor.llm_config['model']
        
        # Example pricing (not actual)
        cost_per_1k_prompt = 0.01
        cost_per_1k_completion = 0.03
        
        if 'gpt-4' in model:
            cost_per_1k_prompt = 0.03
            cost_per_1k_completion = 0.06
        elif 'claude' in model:
            cost_per_1k_prompt = 0.015
            cost_per_1k_completion = 0.075
        
        prompt_cost = (prompt_tokens / 1000) * cost_per_1k_prompt
        completion_cost = (completion_tokens / 1000) * cost_per_1k_completion
        
        return round(prompt_cost + completion_cost, 4)
    
    def _calculate_confidence(
        self, 
        validation: ValidationResult, 
        filtered: FilteredResponse,
        context: SearchResults
    ) -> float:
        """Calculate overall confidence score for the answer."""
        # Base confidence on validation quality
        confidence = validation.quality_score
        
        # Adjust for safety score
        confidence *= filtered.safety_score
        
        # Adjust for context quality
        if context.results:
            avg_relevance = sum(r.relevance_score for r in context.results) / len(context.results)
            confidence *= (0.5 + 0.5 * avg_relevance)  # Context contributes 50%
        else:
            confidence *= 0.5  # No context reduces confidence
        
        # Ensure confidence is between 0 and 1
        return max(0, min(1, confidence))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get answer generation statistics."""
        return {
            'llm_provider': self.config_accessor.llm_config['provider'],
            'llm_model': self.config_accessor.llm_config['model'],
            'response_timeout': self.RESPONSE_TIMEOUT,
            'max_retries': self.MAX_RETRIES,
            'cache_size': len(self._llm_cache)
        }
    
    def shutdown(self):
        """Shutdown the answer generator."""
        self._executor.shutdown(wait=True)
        self.logger.info("Answer generator shutdown complete")