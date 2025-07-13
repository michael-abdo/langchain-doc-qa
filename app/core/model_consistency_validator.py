"""
Embedding Model Consistency Validator - Day 3 Reliability Enhancement
Ensures consistent embedding model usage across all vector operations.

Features:
- Track embedding model versions and configurations
- Detect when different embedding models are used
- Validate embedding characteristics match expected model outputs
- Identify model drift or configuration changes
- Provide migration guidance when models change
- Model compatibility matrix management
"""
import asyncio
import json
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
import pickle

from app.core.common import BaseService, get_service_logger
from app.core.exceptions import ValidationError, VectorStoreError
from app.core.consistency_validator import ValidationReport, ValidationIssue, ValidationSeverity, ValidationType


class ModelValidationType(Enum):
    """Types of model consistency validations."""
    VERSION_MISMATCH = "version_mismatch"
    DIMENSION_MISMATCH = "dimension_mismatch"
    DISTRIBUTION_MISMATCH = "distribution_mismatch"
    ENCODING_MISMATCH = "encoding_mismatch"
    CONFIG_DRIFT = "config_drift"
    COMPATIBILITY = "compatibility"


@dataclass
class EmbeddingModelProfile:
    """Profile of an embedding model's characteristics."""
    model_id: str
    model_name: str
    model_version: str
    embedding_dimension: int
    max_input_length: int
    tokenizer_config: Dict[str, Any] = field(default_factory=dict)
    distribution_stats: Dict[str, float] = field(default_factory=dict)
    config_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    
    def __post_init__(self):
        """Calculate config hash if not provided."""
        if not self.config_hash:
            config_data = {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "dimension": self.embedding_dimension,
                "max_length": self.max_input_length,
                "tokenizer": self.tokenizer_config
            }
            self.config_hash = hashlib.sha256(
                json.dumps(config_data, sort_keys=True).encode()
            ).hexdigest()[:16]


@dataclass
class ModelCompatibilityInfo:
    """Information about model compatibility."""
    source_model_id: str
    target_model_id: str
    is_compatible: bool
    compatibility_score: float  # 0-1, where 1 is fully compatible
    dimension_match: bool
    distribution_similarity: float
    migration_required: bool
    migration_notes: List[str] = field(default_factory=list)
    validation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ModelMigrationPlan:
    """Plan for migrating between embedding models."""
    migration_id: str
    source_model: EmbeddingModelProfile
    target_model: EmbeddingModelProfile
    affected_vector_count: int
    migration_steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_duration_seconds: float = 0.0
    risk_level: str = "low"  # low, medium, high
    backup_required: bool = True
    validation_required: bool = True


class EmbeddingModelConsistencyValidator(BaseService):
    """
    Validates embedding model consistency across vector operations.
    
    Tracks model usage, detects inconsistencies, validates compatibility,
    and provides migration guidance when models change.
    """
    
    def __init__(self):
        super().__init__("model_consistency_validator")
        
        # Model tracking
        self._model_profiles: Dict[str, EmbeddingModelProfile] = {}
        self._active_model_id: Optional[str] = None
        self._model_usage_history: List[Dict[str, Any]] = []
        
        # Compatibility matrix
        self._compatibility_cache: Dict[Tuple[str, str], ModelCompatibilityInfo] = {}
        
        # Validation statistics
        self._validation_stats = {
            "total_validations": 0,
            "model_mismatches_detected": 0,
            "compatibility_checks": 0,
            "migrations_suggested": 0
        }
    
    async def register_embedding_model(
        self,
        model_name: str,
        model_version: str,
        embedding_dimension: int,
        max_input_length: int = 512,
        tokenizer_config: Optional[Dict[str, Any]] = None,
        sample_embeddings: Optional[List[List[float]]] = None
    ) -> EmbeddingModelProfile:
        """
        Register an embedding model profile.
        
        Args:
            model_name: Name of the embedding model
            model_version: Version of the model
            embedding_dimension: Dimension of embeddings
            max_input_length: Maximum input token length
            tokenizer_config: Tokenizer configuration
            sample_embeddings: Sample embeddings for distribution analysis
            
        Returns:
            EmbeddingModelProfile for the registered model
        """
        model_id = f"{model_name}_{model_version}_{embedding_dimension}d"
        
        # Calculate distribution statistics if samples provided
        distribution_stats = {}
        if sample_embeddings:
            distribution_stats = await self._calculate_distribution_stats(sample_embeddings)
        
        profile = EmbeddingModelProfile(
            model_id=model_id,
            model_name=model_name,
            model_version=model_version,
            embedding_dimension=embedding_dimension,
            max_input_length=max_input_length,
            tokenizer_config=tokenizer_config or {},
            distribution_stats=distribution_stats
        )
        
        # Store profile
        self._model_profiles[model_id] = profile
        
        # Set as active if no active model
        if self._active_model_id is None:
            self._active_model_id = model_id
        
        self.logger.info(
            "embedding_model_registered",
            model_id=model_id,
            model_name=model_name,
            version=model_version,
            dimension=embedding_dimension,
            is_active=self._active_model_id == model_id
        )
        
        return profile
    
    async def validate_embedding_consistency(
        self,
        embeddings: List[List[float]],
        expected_model_id: Optional[str] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> ValidationReport:
        """
        Validate embedding consistency with expected model.
        
        Args:
            embeddings: List of embeddings to validate
            expected_model_id: Expected model ID (uses active if not provided)
            metadata: Optional metadata containing model information
            
        Returns:
            ValidationReport with consistency validation results
        """
        validation_id = f"model_consistency_{int(datetime.now().timestamp())}"
        start_time = asyncio.get_event_loop().time()
        
        if expected_model_id is None:
            expected_model_id = self._active_model_id
        
        if expected_model_id is None:
            raise ValidationError("No embedding model registered or specified")
        
        expected_profile = self._model_profiles.get(expected_model_id)
        if not expected_profile:
            raise ValidationError(f"Unknown model ID: {expected_model_id}")
        
        self.logger.info(
            "embedding_consistency_validation_started",
            validation_id=validation_id,
            embedding_count=len(embeddings),
            expected_model=expected_model_id
        )
        
        issues = []
        total_checks = 0
        passed_checks = 0
        
        # Check 1: Dimension consistency
        total_checks += 1
        dimension_issues = await self._validate_dimensions(embeddings, expected_profile)
        if dimension_issues:
            issues.extend(dimension_issues)
        else:
            passed_checks += 1
        
        # Check 2: Distribution consistency
        total_checks += 1
        distribution_issues = await self._validate_distribution(embeddings, expected_profile)
        if distribution_issues:
            issues.extend(distribution_issues)
        else:
            passed_checks += 1
        
        # Check 3: Model metadata consistency (if provided)
        if metadata:
            total_checks += 1
            metadata_issues = await self._validate_model_metadata(metadata, expected_profile)
            if metadata_issues:
                issues.extend(metadata_issues)
            else:
                passed_checks += 1
        
        # Check 4: Encoding characteristics
        total_checks += 1
        encoding_issues = await self._validate_encoding_characteristics(embeddings, expected_profile)
        if encoding_issues:
            issues.extend(encoding_issues)
        else:
            passed_checks += 1
        
        # Update model usage
        expected_profile.usage_count += 1
        expected_profile.last_seen = datetime.now().isoformat()
        
        # Record usage history
        self._model_usage_history.append({
            "timestamp": datetime.now().isoformat(),
            "model_id": expected_model_id,
            "embedding_count": len(embeddings),
            "validation_passed": len(issues) == 0
        })
        
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        report = ValidationReport(
            validation_id=validation_id,
            timestamp=datetime.now().isoformat(),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=total_checks - passed_checks,
            issues=issues,
            summary={
                "expected_model": expected_model_id,
                "embedding_count": len(embeddings),
                "model_profile": asdict(expected_profile)
            },
            performance_metrics={
                "validation_time_seconds": elapsed_time,
                "embeddings_per_second": len(embeddings) / elapsed_time if elapsed_time > 0 else 0
            }
        )
        
        self._validation_stats["total_validations"] += 1
        if issues:
            self._validation_stats["model_mismatches_detected"] += 1
        
        self.logger.info(
            "embedding_consistency_validation_completed",
            validation_id=validation_id,
            success_rate=report.success_rate,
            issues_found=len(issues)
        )
        
        return report
    
    async def check_model_compatibility(
        self,
        source_model_id: str,
        target_model_id: str,
        sample_embeddings: Optional[Dict[str, List[List[float]]]] = None
    ) -> ModelCompatibilityInfo:
        """
        Check compatibility between two embedding models.
        
        Args:
            source_model_id: Source model ID
            target_model_id: Target model ID
            sample_embeddings: Optional sample embeddings from both models
            
        Returns:
            ModelCompatibilityInfo with compatibility analysis
        """
        cache_key = (source_model_id, target_model_id)
        
        # Check cache first
        if cache_key in self._compatibility_cache:
            cached_info = self._compatibility_cache[cache_key]
            # Invalidate cache after 24 hours
            cache_time = datetime.fromisoformat(cached_info.validation_timestamp)
            if datetime.now() - cache_time < timedelta(hours=24):
                return cached_info
        
        source_profile = self._model_profiles.get(source_model_id)
        target_profile = self._model_profiles.get(target_model_id)
        
        if not source_profile or not target_profile:
            raise ValidationError(f"Unknown model IDs: {source_model_id} or {target_model_id}")
        
        self.logger.info(
            "model_compatibility_check_started",
            source_model=source_model_id,
            target_model=target_model_id
        )
        
        # Check dimension compatibility
        dimension_match = source_profile.embedding_dimension == target_profile.embedding_dimension
        
        # Calculate distribution similarity
        distribution_similarity = await self._calculate_distribution_similarity(
            source_profile.distribution_stats,
            target_profile.distribution_stats
        )
        
        # Determine overall compatibility
        is_compatible = dimension_match and distribution_similarity > 0.8
        compatibility_score = (
            (1.0 if dimension_match else 0.0) * 0.5 +
            distribution_similarity * 0.5
        )
        
        # Determine if migration is required
        migration_required = not dimension_match or distribution_similarity < 0.9
        
        # Generate migration notes
        migration_notes = []
        if not dimension_match:
            migration_notes.append(
                f"Dimension mismatch: {source_profile.embedding_dimension} ’ "
                f"{target_profile.embedding_dimension}. Full re-encoding required."
            )
        
        if distribution_similarity < 0.9:
            migration_notes.append(
                f"Distribution characteristics differ (similarity: {distribution_similarity:.2f}). "
                "Search quality may be affected."
            )
        
        if source_profile.model_name != target_profile.model_name:
            migration_notes.append(
                f"Different model families: {source_profile.model_name} ’ "
                f"{target_profile.model_name}. Thorough testing recommended."
            )
        
        compatibility_info = ModelCompatibilityInfo(
            source_model_id=source_model_id,
            target_model_id=target_model_id,
            is_compatible=is_compatible,
            compatibility_score=compatibility_score,
            dimension_match=dimension_match,
            distribution_similarity=distribution_similarity,
            migration_required=migration_required,
            migration_notes=migration_notes
        )
        
        # Cache result
        self._compatibility_cache[cache_key] = compatibility_info
        self._validation_stats["compatibility_checks"] += 1
        
        self.logger.info(
            "model_compatibility_check_completed",
            source_model=source_model_id,
            target_model=target_model_id,
            is_compatible=is_compatible,
            score=compatibility_score
        )
        
        return compatibility_info
    
    async def generate_migration_plan(
        self,
        source_model_id: str,
        target_model_id: str,
        total_vectors: int
    ) -> ModelMigrationPlan:
        """
        Generate a migration plan for transitioning between models.
        
        Args:
            source_model_id: Current model ID
            target_model_id: Target model ID
            total_vectors: Total number of vectors to migrate
            
        Returns:
            ModelMigrationPlan with migration steps
        """
        source_profile = self._model_profiles.get(source_model_id)
        target_profile = self._model_profiles.get(target_model_id)
        
        if not source_profile or not target_profile:
            raise ValidationError(f"Unknown model IDs: {source_model_id} or {target_model_id}")
        
        # Check compatibility
        compatibility = await self.check_model_compatibility(source_model_id, target_model_id)
        
        migration_id = f"migration_{source_model_id}_to_{target_model_id}_{int(datetime.now().timestamp())}"
        
        # Determine migration steps
        migration_steps = []
        
        # Step 1: Backup
        migration_steps.append({
            "step": 1,
            "action": "backup_vectors",
            "description": "Create full backup of existing vector store",
            "estimated_duration": total_vectors * 0.001  # ~1ms per vector
        })
        
        # Step 2: Validation
        migration_steps.append({
            "step": 2,
            "action": "validate_source",
            "description": "Validate source vector integrity",
            "estimated_duration": total_vectors * 0.0005  # ~0.5ms per vector
        })
        
        # Step 3: Re-encode (if needed)
        if not compatibility.dimension_match:
            migration_steps.append({
                "step": 3,
                "action": "re_encode_documents",
                "description": f"Re-encode all documents with {target_profile.model_name}",
                "estimated_duration": total_vectors * 0.1  # ~100ms per vector
            })
        
        # Step 4: Batch migration
        batch_size = 1000
        num_batches = (total_vectors + batch_size - 1) // batch_size
        migration_steps.append({
            "step": len(migration_steps) + 1,
            "action": "batch_migration",
            "description": f"Migrate vectors in {num_batches} batches of {batch_size}",
            "estimated_duration": total_vectors * 0.01,  # ~10ms per vector
            "batch_config": {
                "batch_size": batch_size,
                "num_batches": num_batches,
                "parallel_batches": min(4, num_batches)
            }
        })
        
        # Step 5: Validation
        migration_steps.append({
            "step": len(migration_steps) + 1,
            "action": "validate_target",
            "description": "Validate migrated vectors",
            "estimated_duration": total_vectors * 0.0005
        })
        
        # Step 6: Update metadata
        migration_steps.append({
            "step": len(migration_steps) + 1,
            "action": "update_metadata",
            "description": "Update vector metadata with new model information",
            "estimated_duration": total_vectors * 0.0002
        })
        
        # Calculate total duration
        total_duration = sum(step.get("estimated_duration", 0) for step in migration_steps)
        
        # Determine risk level
        risk_level = "low"
        if not compatibility.dimension_match:
            risk_level = "high"
        elif compatibility.compatibility_score < 0.8:
            risk_level = "medium"
        
        plan = ModelMigrationPlan(
            migration_id=migration_id,
            source_model=source_profile,
            target_model=target_profile,
            affected_vector_count=total_vectors,
            migration_steps=migration_steps,
            estimated_duration_seconds=total_duration,
            risk_level=risk_level,
            backup_required=True,
            validation_required=True
        )
        
        self._validation_stats["migrations_suggested"] += 1
        
        self.logger.info(
            "migration_plan_generated",
            migration_id=migration_id,
            source_model=source_model_id,
            target_model=target_model_id,
            risk_level=risk_level,
            estimated_duration=total_duration
        )
        
        return plan
    
    async def _validate_dimensions(
        self,
        embeddings: List[List[float]],
        expected_profile: EmbeddingModelProfile
    ) -> List[ValidationIssue]:
        """Validate embedding dimensions match expected model."""
        issues = []
        
        if not embeddings:
            return issues
        
        expected_dim = expected_profile.embedding_dimension
        mismatched_dims = []
        
        for i, embedding in enumerate(embeddings):
            actual_dim = len(embedding)
            if actual_dim != expected_dim:
                mismatched_dims.append({
                    "index": i,
                    "expected": expected_dim,
                    "actual": actual_dim
                })
        
        if mismatched_dims:
            issues.append(ValidationIssue(
                type=ValidationType.DIMENSION,
                severity=ValidationSeverity.CRITICAL,
                message=f"Embedding dimensions don't match model {expected_profile.model_id}",
                details={
                    "expected_dimension": expected_dim,
                    "mismatched_count": len(mismatched_dims),
                    "sample_mismatches": mismatched_dims[:10]
                },
                affected_items=[str(m["index"]) for m in mismatched_dims],
                remediation_suggestion=f"Re-encode with {expected_profile.model_name} v{expected_profile.model_version}"
            ))
        
        return issues
    
    async def _validate_distribution(
        self,
        embeddings: List[List[float]],
        expected_profile: EmbeddingModelProfile
    ) -> List[ValidationIssue]:
        """Validate embedding distribution matches expected model."""
        issues = []
        
        if not embeddings or not expected_profile.distribution_stats:
            return issues
        
        # Calculate current distribution stats
        current_stats = await self._calculate_distribution_stats(embeddings)
        expected_stats = expected_profile.distribution_stats
        
        # Compare distributions
        significant_differences = []
        
        for stat_name in ["mean_norm", "std_norm", "mean_mean", "std_mean"]:
            if stat_name in current_stats and stat_name in expected_stats:
                current_val = current_stats[stat_name]
                expected_val = expected_stats[stat_name]
                
                # Calculate relative difference
                if expected_val != 0:
                    rel_diff = abs((current_val - expected_val) / expected_val)
                    if rel_diff > 0.2:  # More than 20% difference
                        significant_differences.append({
                            "statistic": stat_name,
                            "current": current_val,
                            "expected": expected_val,
                            "relative_difference": rel_diff
                        })
        
        if significant_differences:
            issues.append(ValidationIssue(
                type=ValidationType.QUALITY,
                severity=ValidationSeverity.WARNING,
                message="Embedding distribution doesn't match expected model characteristics",
                details={
                    "model_id": expected_profile.model_id,
                    "significant_differences": significant_differences,
                    "current_stats": current_stats,
                    "expected_stats": expected_stats
                },
                remediation_suggestion="Verify correct model is being used for encoding"
            ))
        
        return issues
    
    async def _validate_model_metadata(
        self,
        metadata: List[Dict[str, Any]],
        expected_profile: EmbeddingModelProfile
    ) -> List[ValidationIssue]:
        """Validate model information in metadata."""
        issues = []
        
        model_mismatches = []
        
        for i, meta in enumerate(metadata):
            if "model_name" in meta:
                if meta["model_name"] != expected_profile.model_name:
                    model_mismatches.append({
                        "index": i,
                        "metadata_model": meta.get("model_name"),
                        "expected_model": expected_profile.model_name
                    })
            
            if "model_version" in meta:
                if meta["model_version"] != expected_profile.model_version:
                    model_mismatches.append({
                        "index": i,
                        "metadata_version": meta.get("model_version"),
                        "expected_version": expected_profile.model_version
                    })
        
        if model_mismatches:
            issues.append(ValidationIssue(
                type=ValidationType.ALIGNMENT,
                severity=ValidationSeverity.ERROR,
                message="Model information in metadata doesn't match expected model",
                details={
                    "expected_model": expected_profile.model_id,
                    "mismatch_count": len(model_mismatches),
                    "sample_mismatches": model_mismatches[:10]
                },
                affected_items=[str(m["index"]) for m in model_mismatches],
                remediation_suggestion="Update metadata or re-encode with correct model"
            ))
        
        return issues
    
    async def _validate_encoding_characteristics(
        self,
        embeddings: List[List[float]],
        expected_profile: EmbeddingModelProfile
    ) -> List[ValidationIssue]:
        """Validate encoding characteristics match expected model."""
        issues = []
        
        if not embeddings:
            return issues
        
        # Convert to numpy for analysis
        try:
            embedding_array = np.array(embeddings, dtype=np.float32)
        except Exception:
            return issues
        
        # Check for model-specific characteristics
        # Example: Some models produce normalized embeddings
        norms = np.linalg.norm(embedding_array, axis=1)
        
        # Check if embeddings appear to be normalized (norm H 1.0)
        normalized_count = np.sum(np.abs(norms - 1.0) < 0.01)
        normalization_ratio = normalized_count / len(embeddings)
        
        # Some models always normalize, others don't
        if expected_profile.model_name in ["sentence-transformers", "all-MiniLM-L6-v2"]:
            # These models typically produce normalized embeddings
            if normalization_ratio < 0.9:
                issues.append(ValidationIssue(
                    type=ValidationType.QUALITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Embeddings don't appear to be normalized as expected for {expected_profile.model_name}",
                    details={
                        "normalization_ratio": normalization_ratio,
                        "expected_normalized": True,
                        "norm_statistics": {
                            "mean": float(np.mean(norms)),
                            "std": float(np.std(norms)),
                            "min": float(np.min(norms)),
                            "max": float(np.max(norms))
                        }
                    },
                    remediation_suggestion="Verify correct model configuration and preprocessing"
                ))
        
        return issues
    
    async def _calculate_distribution_stats(
        self,
        embeddings: List[List[float]]
    ) -> Dict[str, float]:
        """Calculate distribution statistics for embeddings."""
        if not embeddings:
            return {}
        
        try:
            embedding_array = np.array(embeddings, dtype=np.float32)
            
            # Calculate various statistics
            norms = np.linalg.norm(embedding_array, axis=1)
            means = np.mean(embedding_array, axis=1)
            
            stats = {
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "min_norm": float(np.min(norms)),
                "max_norm": float(np.max(norms)),
                "mean_mean": float(np.mean(means)),
                "std_mean": float(np.std(means)),
                "sparsity": float(np.mean(embedding_array == 0)),
                "dimension": embedding_array.shape[1]
            }
            
            return stats
            
        except Exception as e:
            self.logger.warning(
                "distribution_stats_calculation_failed",
                error=str(e)
            )
            return {}
    
    async def _calculate_distribution_similarity(
        self,
        stats1: Dict[str, float],
        stats2: Dict[str, float]
    ) -> float:
        """Calculate similarity between two distribution statistics."""
        if not stats1 or not stats2:
            return 0.0
        
        # Compare key statistics
        similarities = []
        
        for key in ["mean_norm", "std_norm", "mean_mean", "std_mean"]:
            if key in stats1 and key in stats2:
                val1 = stats1[key]
                val2 = stats2[key]
                
                # Calculate similarity (1 - relative difference)
                if val1 != 0 or val2 != 0:
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        similarity = 1.0 - abs(val1 - val2) / max_val
                        similarities.append(max(0, similarity))
        
        if similarities:
            return sum(similarities) / len(similarities)
        
        return 0.0
    
    # Public API methods
    
    def set_active_model(self, model_id: str) -> None:
        """Set the active embedding model."""
        if model_id not in self._model_profiles:
            raise ValidationError(f"Unknown model ID: {model_id}")
        
        self._active_model_id = model_id
        self.logger.info("active_embedding_model_set", model_id=model_id)
    
    def get_active_model(self) -> Optional[EmbeddingModelProfile]:
        """Get the active embedding model profile."""
        if self._active_model_id:
            return self._model_profiles.get(self._active_model_id)
        return None
    
    def list_registered_models(self) -> List[EmbeddingModelProfile]:
        """List all registered embedding models."""
        return list(self._model_profiles.values())
    
    def get_model_usage_stats(self) -> Dict[str, Any]:
        """Get model usage statistics."""
        model_usage = defaultdict(int)
        
        for entry in self._model_usage_history:
            model_usage[entry["model_id"]] += entry["embedding_count"]
        
        return {
            "registered_models": len(self._model_profiles),
            "active_model": self._active_model_id,
            "total_validations": self._validation_stats["total_validations"],
            "model_usage": dict(model_usage),
            "validation_stats": self._validation_stats
        }


# Global instance for shared use
model_consistency_validator = EmbeddingModelConsistencyValidator()