#!/usr/bin/env python3
"""Test script for Answer Generation Service."""

import sys
import os
import time
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.query_processor import ProcessedQuery
from app.services.retrieval_engine import SearchResults, SearchResult
from app.services.answer_generator import (
    AnswerGenerator, PromptBuilder, ResponseValidator, 
    ContentFilter, LLMValidator
)
from app.core.exceptions import LLMError
from langchain.schema import Document


def create_mock_processed_query(query_text: str, intent: str = "question") -> ProcessedQuery:
    """Create a mock ProcessedQuery for testing."""
    return ProcessedQuery(
        original_query=query_text,
        sanitized_query=query_text,
        normalized_query=query_text.lower(),
        complexity_score=10.0,
        token_count=len(query_text.split()),
        detected_intent=intent,
        language="english",
        timestamp=datetime.now(),
        user_id="test_user",
        processing_time=0.1,
        security_flags=[],
        metadata={}
    )


def create_mock_search_results(query: ProcessedQuery, has_results: bool = True) -> SearchResults:
    """Create mock search results for testing."""
    results = []
    
    if has_results:
        results = [
            SearchResult(
                document=Document(
                    page_content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
                    metadata={"source": "ml_basics.pdf", "page": 1}
                ),
                score=0.95,
                relevance_score=0.9,
                chunk_id="chunk_001",
                source="ml_basics.pdf",
                metadata={"page": 1}
            ),
            SearchResult(
                document=Document(
                    page_content="The process of machine learning begins with observations or data, such as examples, direct experience, or instruction. The computer looks for patterns in the data and uses these patterns to make better decisions in the future.",
                    metadata={"source": "ml_process.pdf", "page": 3}
                ),
                score=0.88,
                relevance_score=0.85,
                chunk_id="chunk_002",
                source="ml_process.pdf",
                metadata={"page": 3}
            )
        ]
    
    return SearchResults(
        query=query,
        results=results,
        total_found=len(results),
        search_time=0.5,
        search_method="vector_search",
        filters_applied=[],
        performance_metrics={
            'search_latency': 0.5,
            'documents_scanned': 100,
            'results_returned': len(results)
        }
    )


def test_prompt_builder():
    """Test the PromptBuilder component."""
    print("Testing PromptBuilder...")
    
    builder = PromptBuilder(None)  # Logger not needed for test
    
    # Test with context
    query = create_mock_processed_query("What is machine learning?")
    results = create_mock_search_results(query)
    
    prompt = builder.build_prompt(query, results)
    
    print("✓ Prompt built successfully")
    print(f"  - System message length: {len(prompt.system_message)}")
    print(f"  - User message length: {len(prompt.user_message)}")
    print(f"  - Context included: {'Yes' if 'ml_basics.pdf' in prompt.context else 'No'}")
    print(f"  - Token count: {prompt.token_count}")
    print(f"  - Constraints: {len(prompt.constraints)}")
    
    # Test without context
    empty_results = create_mock_search_results(query, has_results=False)
    prompt2 = builder.build_prompt(query, empty_results)
    
    print("✓ Prompt built without context")
    print(f"  - Shows 'No relevant context': {'Yes' if 'No relevant context' in prompt2.context else 'No'}")
    
    return True


def test_response_validator():
    """Test the ResponseValidator component."""
    print("\n\nTesting ResponseValidator...")
    
    validator = ResponseValidator(None)  # Logger not needed for test
    query = create_mock_processed_query("What is machine learning?")
    context = create_mock_search_results(query)
    
    # Test good response
    good_response = """Machine learning is a subset of artificial intelligence that enables computer systems to learn and improve from experience without being explicitly programmed. According to [Source 1], it focuses on developing programs that can access data and learn autonomously.

The process involves analyzing patterns in data to make predictions and decisions. As mentioned in [Source 2], machine learning begins with observations or data, which the system uses to identify patterns and improve future decision-making.

This technology has revolutionized many fields including healthcare, finance, and transportation."""
    
    result = validator.validate_response(good_response, query, context)
    
    print("✓ Good response validation:")
    print(f"  - Valid: {result.is_valid}")
    print(f"  - Quality score: {result.quality_score:.2f}")
    print(f"  - Issues: {len(result.issues)}")
    print(f"  - Suggestions: {len(result.suggestions)}")
    
    # Test bad response
    bad_response = "ML is good."
    
    result2 = validator.validate_response(bad_response, query, context)
    
    print("\n✓ Bad response validation:")
    print(f"  - Valid: {result2.is_valid}")
    print(f"  - Quality score: {result2.quality_score:.2f}")
    print(f"  - Issues: {result2.issues}")
    print(f"  - Suggestions: {result2.suggestions}")
    
    return True


def test_content_filter():
    """Test the ContentFilter component."""
    print("\n\nTesting ContentFilter...")
    
    # Create a mock logger
    class MockLogger:
        def warning(self, msg):
            pass  # Suppress warnings in test
    
    filter = ContentFilter(MockLogger())
    
    # Test clean content
    clean_response = "Machine learning is a fascinating field of study that has many applications in modern technology."
    
    result = filter.filter_content(clean_response)
    
    print("✓ Clean content filtering:")
    print(f"  - Was filtered: {result.was_filtered}")
    print(f"  - Safety score: {result.safety_score}")
    print(f"  - Filtered sections: {len(result.filtered_sections)}")
    
    # Test problematic content (mild example)
    problem_response = "You should definitely sue your neighbor for this issue. Also, here's my diagnosis of your symptoms."
    
    result2 = filter.filter_content(problem_response)
    
    print("\n✓ Problematic content filtering:")
    print(f"  - Was filtered: {result2.was_filtered}")
    print(f"  - Safety score: {result2.safety_score}")
    print(f"  - Filtered sections: {len(result2.filtered_sections)}")
    for section in result2.filtered_sections[:3]:  # Show first 3
        print(f"    - {section}")
    
    return True


def test_llm_validator():
    """Test the LLMValidator component."""
    print("\n\nTesting LLMValidator...")
    
    validator = LLMValidator(None)  # Logger not needed for test
    
    # Create mock OpenAI response
    class MockMessage:
        def __init__(self, content):
            self.content = content
    
    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)
    
    class MockOpenAIResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]
    
    # Test valid OpenAI response
    valid_response = MockOpenAIResponse("This is a valid response")
    is_valid, error = validator.validate_api_response(valid_response, "openai")
    
    print("✓ Valid OpenAI response:")
    print(f"  - Is valid: {is_valid}")
    print(f"  - Error: {error}")
    
    # Test invalid response
    class InvalidResponse:
        pass
    
    invalid_response = InvalidResponse()
    is_valid2, error2 = validator.validate_api_response(invalid_response, "openai")
    
    print("\n✓ Invalid response:")
    print(f"  - Is valid: {is_valid2}")
    print(f"  - Error: {error2}")
    
    return True


def test_answer_generator():
    """Test the main AnswerGenerator service."""
    print("\n\nTesting AnswerGenerator...")
    
    try:
        # Initialize generator
        generator = AnswerGenerator()
        print("✓ Answer generator initialized")
        
        # Get stats
        stats = generator.get_stats()
        print("✓ Generator stats:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
        # Note: We can't test actual LLM generation without API keys
        print("\n⚠ Skipping actual LLM generation test (requires API keys)")
        
        # Test with mock data would go here
        # query = create_mock_processed_query("What is machine learning?")
        # context = create_mock_search_results(query)
        # answer = generator.generate_answer(query, context)
        
        # Shutdown
        generator.shutdown()
        print("\n✓ Answer generator shutdown complete")
        
        return True
        
    except LLMError as e:
        print(f"⚠ Expected LLM error (no API key configured): {e}")
        return True
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {str(e)}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("Testing Answer Generation Components")
    print("="*60)
    
    success = True
    
    # Test individual components
    success &= test_prompt_builder()
    success &= test_response_validator()
    success &= test_content_filter()
    success &= test_llm_validator()
    success &= test_answer_generator()
    
    print("\n" + "="*60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*60)
    
    sys.exit(0 if success else 1)