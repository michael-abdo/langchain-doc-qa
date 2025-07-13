"""
Vector-Metadata Consistency Validator - Day 3 Reliability Enhancement
Provides comprehensive consistency validation for vector store operations.

Features:
- Vector-metadata count and alignment validation
- Dimension consistency validation across batches
- Embedding quality validation (NaN/infinity detection)
- ID uniqueness validation across operations
- Cross-component reference validation
- Comprehensive validation reporting with severity levels
"""
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import math
import json

from app.core.common import BaseService, get_service_logger, datetime
from app.core.exceptions import ValidationError, VectorStoreError
from app.core.vector_reliability import memory_manager, performance_monitor


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Types of consistency validations."""
    ALIGNMENT = "alignment"
    DIMENSION = "dimension"
    QUALITY = "quality"
    UNIQUENESS = "uniqueness"
    REFERENCE = "reference"
    INTEGRITY = "integrity"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    type: ValidationType
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    affected_items: List[str] = field(default_factory=list)
    remediation_suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    validation_id: str
    timestamp: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_checks == 0:
            return 100.0
        return (self.passed_checks / self.total_checks) * 100.0
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if report contains critical issues."""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    @property
    def severity_counts(self) -> Dict[str, int]:
        """Get count of issues by severity."""
        counts = Counter(issue.severity.value for issue in self.issues)
        return dict(counts)


class VectorConsistencyValidator(BaseService):
    """
    Comprehensive validator for vector store consistency and integrity.
    
    Provides validation for vector-metadata alignment, dimension consistency,
    embedding quality, ID uniqueness, and cross-component references.
    """
    
    def __init__(self):
        super().__init__("consistency_validator")
        self._validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "critical_issues_found": 0
        }
    
    async def validate_vector_metadata_alignment(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        document_ids: Optional[List[str]] = None
    ) -> ValidationReport:
        """
        Validate alignment between vectors, metadata, and document IDs.
        
        Args:
            vectors: List of embedding vectors
            metadatas: List of metadata dictionaries
            document_ids: Optional list of document IDs
            
        Returns:
            ValidationReport with alignment validation results
        """
        validation_id = f"alignment_{int(datetime.now().timestamp())}"
        start_time = asyncio.get_event_loop().time()
        
        self.logger.info(
            "vector_metadata_alignment_validation_started",
            validation_id=validation_id,
            vector_count=len(vectors),
            metadata_count=len(metadatas),
            has_document_ids=document_ids is not None
        )
        
        # Memory check before processing
        memory_manager.enforce_memory_limits("alignment_validation")
        
        issues = []
        total_checks = 0
        passed_checks = 0
        
        async with performance_monitor.track_operation("alignment_validation", timeout=60.0):
            # Check 1: Vector-metadata count alignment
            total_checks += 1
            if len(vectors) != len(metadatas):
                issues.append(ValidationIssue(
                    type=ValidationType.ALIGNMENT,
                    severity=ValidationSeverity.CRITICAL,
                    message="Vector and metadata counts do not match",
                    details={
                        "vector_count": len(vectors),
                        "metadata_count": len(metadatas),
                        "difference": abs(len(vectors) - len(metadatas))
                    },
                    remediation_suggestion="Ensure vector and metadata lists have identical lengths"
                ))
            else:
                passed_checks += 1
            
            # Check 2: Document ID alignment (if provided)
            if document_ids is not None:
                total_checks += 1
                if len(vectors) != len(document_ids):
                    issues.append(ValidationIssue(
                        type=ValidationType.ALIGNMENT,
                        severity=ValidationSeverity.CRITICAL,
                        message="Vector and document ID counts do not match",
                        details={
                            "vector_count": len(vectors),
                            "document_id_count": len(document_ids),
                            "difference": abs(len(vectors) - len(document_ids))
                        },
                        remediation_suggestion="Ensure document ID list matches vector count"
                    ))
                else:
                    passed_checks += 1
            
            # Check 3: Metadata structure consistency
            total_checks += 1
            metadata_structure_issues = await self._validate_metadata_structure(metadatas)
            if metadata_structure_issues:
                issues.extend(metadata_structure_issues)
            else:
                passed_checks += 1
            
            # Check 4: Vector-metadata field alignment
            total_checks += 1
            field_alignment_issues = await self._validate_field_alignment(vectors, metadatas)
            if field_alignment_issues:
                issues.extend(field_alignment_issues)
            else:
                passed_checks += 1
        
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        report = ValidationReport(
            validation_id=validation_id,
            timestamp=datetime.now().isoformat(),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=total_checks - passed_checks,
            issues=issues,
            summary={
                "vector_count": len(vectors),
                "metadata_count": len(metadatas),
                "document_id_count": len(document_ids) if document_ids else 0,
                "alignment_status": "aligned" if len(vectors) == len(metadatas) else "misaligned"
            },
            performance_metrics={
                "validation_time_seconds": elapsed_time,
                "items_per_second": len(vectors) / elapsed_time if elapsed_time > 0 else 0
            }
        )
        
        await self._update_validation_stats(report)
        
        self.logger.info(
            "vector_metadata_alignment_validation_completed",
            validation_id=validation_id,
            success_rate=report.success_rate,
            critical_issues=report.has_critical_issues,
            total_issues=len(issues)
        )
        
        return report
    
    async def validate_dimension_consistency(
        self,
        vectors: List[List[float]],
        expected_dimension: Optional[int] = None
    ) -> ValidationReport:
        """
        Validate dimension consistency across vector batches.
        
        Args:
            vectors: List of embedding vectors
            expected_dimension: Expected vector dimension
            
        Returns:
            ValidationReport with dimension validation results
        """
        validation_id = f"dimension_{int(datetime.now().timestamp())}"
        start_time = asyncio.get_event_loop().time()
        
        self.logger.info(
            "dimension_consistency_validation_started",
            validation_id=validation_id,
            vector_count=len(vectors),
            expected_dimension=expected_dimension
        )
        
        # Memory check
        memory_manager.enforce_memory_limits("dimension_validation")
        
        issues = []
        total_checks = 0
        passed_checks = 0
        
        if not vectors:
            return ValidationReport(
                validation_id=validation_id,
                timestamp=datetime.now().isoformat(),
                total_checks=1,
                passed_checks=1,
                failed_checks=0,
                summary={"status": "empty_vector_list"}
            )
        
        async with performance_monitor.track_operation("dimension_validation", timeout=30.0):
            # Determine expected dimension
            if expected_dimension is None:
                expected_dimension = len(vectors[0]) if vectors else 0
            
            dimension_counts = defaultdict(int)
            inconsistent_vectors = []
            
            # Check each vector's dimension
            for i, vector in enumerate(vectors):
                total_checks += 1
                vector_dimension = len(vector)
                dimension_counts[vector_dimension] += 1
                
                if vector_dimension != expected_dimension:
                    inconsistent_vectors.append({
                        "index": i,
                        "expected": expected_dimension,
                        "actual": vector_dimension
                    })
                else:
                    passed_checks += 1
            
            # Report dimension inconsistencies
            if inconsistent_vectors:
                issues.append(ValidationIssue(
                    type=ValidationType.DIMENSION,
                    severity=ValidationSeverity.ERROR,
                    message=f"Found {len(inconsistent_vectors)} vectors with inconsistent dimensions",
                    details={
                        "expected_dimension": expected_dimension,
                        "dimension_distribution": dict(dimension_counts),
                        "inconsistent_count": len(inconsistent_vectors),
                        "sample_inconsistencies": inconsistent_vectors[:10]  # Limit sample size
                    },
                    affected_items=[str(v["index"]) for v in inconsistent_vectors[:100]],
                    remediation_suggestion="Re-encode affected vectors with correct embedding model"
                ))
            
            # Check for dimension variety (warning if too many different dimensions)
            unique_dimensions = len(dimension_counts)
            if unique_dimensions > 5:
                issues.append(ValidationIssue(
                    type=ValidationType.DIMENSION,
                    severity=ValidationSeverity.WARNING,
                    message=f"High dimension variety detected: {unique_dimensions} different dimensions",
                    details={
                        "dimension_distribution": dict(dimension_counts),
                        "most_common_dimension": max(dimension_counts.keys(), key=dimension_counts.get)
                    },
                    remediation_suggestion="Consider standardizing embedding model across all vectors"
                ))
        
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        report = ValidationReport(
            validation_id=validation_id,
            timestamp=datetime.now().isoformat(),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=total_checks - passed_checks,
            issues=issues,
            summary={
                "vector_count": len(vectors),
                "expected_dimension": expected_dimension,
                "dimension_distribution": dict(dimension_counts),
                "consistent_vectors": passed_checks,
                "inconsistent_vectors": len(inconsistent_vectors)
            },
            performance_metrics={
                "validation_time_seconds": elapsed_time,
                "vectors_per_second": len(vectors) / elapsed_time if elapsed_time > 0 else 0
            }
        )
        
        await self._update_validation_stats(report)
        
        self.logger.info(
            "dimension_consistency_validation_completed",
            validation_id=validation_id,
            success_rate=report.success_rate,
            inconsistent_count=len(inconsistent_vectors)
        )
        
        return report
    
    async def validate_embedding_quality(
        self,
        vectors: List[List[float]],
        check_outliers: bool = True
    ) -> ValidationReport:
        """
        Validate embedding quality (NaN/infinity detection, outlier analysis).
        
        Args:
            vectors: List of embedding vectors
            check_outliers: Whether to perform outlier detection
            
        Returns:
            ValidationReport with quality validation results
        """
        validation_id = f"quality_{int(datetime.now().timestamp())}"
        start_time = asyncio.get_event_loop().time()
        
        self.logger.info(
            "embedding_quality_validation_started",
            validation_id=validation_id,
            vector_count=len(vectors),
            check_outliers=check_outliers
        )
        
        # Memory check
        memory_manager.enforce_memory_limits("quality_validation")
        
        issues = []
        total_checks = 0
        passed_checks = 0
        
        if not vectors:
            return ValidationReport(
                validation_id=validation_id,
                timestamp=datetime.now().isoformat(),
                total_checks=1,
                passed_checks=1,
                failed_checks=0,
                summary={"status": "empty_vector_list"}
            )
        
        async with performance_monitor.track_operation("quality_validation", timeout=120.0):
            # Convert to numpy for efficient processing
            try:
                vector_array = np.array(vectors, dtype=np.float32)
            except Exception as e:
                issues.append(ValidationIssue(
                    type=ValidationType.QUALITY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Failed to convert vectors to numpy array: {str(e)}",
                    details={"error": str(e)},
                    remediation_suggestion="Check vector data types and ensure all vectors are numeric"
                ))
                total_checks += 1
                
                return ValidationReport(
                    validation_id=validation_id,
                    timestamp=datetime.now().isoformat(),
                    total_checks=total_checks,
                    passed_checks=0,
                    failed_checks=1,
                    issues=issues
                )
            
            # Check 1: NaN detection
            total_checks += 1
            nan_mask = np.isnan(vector_array)
            nan_vectors = np.any(nan_mask, axis=1)
            nan_count = np.sum(nan_vectors)
            
            if nan_count > 0:
                nan_indices = np.where(nan_vectors)[0].tolist()
                issues.append(ValidationIssue(
                    type=ValidationType.QUALITY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Found {nan_count} vectors containing NaN values",
                    details={
                        "nan_vector_count": int(nan_count),
                        "total_nan_values": int(np.sum(nan_mask)),
                        "affected_vector_indices": nan_indices[:50]  # Limit sample size
                    },
                    affected_items=[str(i) for i in nan_indices],
                    remediation_suggestion="Re-encode vectors with NaN values"
                ))
            else:
                passed_checks += 1
            
            # Check 2: Infinity detection
            total_checks += 1
            inf_mask = np.isinf(vector_array)
            inf_vectors = np.any(inf_mask, axis=1)
            inf_count = np.sum(inf_vectors)
            
            if inf_count > 0:
                inf_indices = np.where(inf_vectors)[0].tolist()
                issues.append(ValidationIssue(
                    type=ValidationType.QUALITY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Found {inf_count} vectors containing infinite values",
                    details={
                        "inf_vector_count": int(inf_count),
                        "total_inf_values": int(np.sum(inf_mask)),
                        "affected_vector_indices": inf_indices[:50]
                    },
                    affected_items=[str(i) for i in inf_indices],
                    remediation_suggestion="Re-encode vectors with infinite values"
                ))
            else:
                passed_checks += 1
            
            # Check 3: Zero vectors
            total_checks += 1
            zero_vectors = np.all(vector_array == 0, axis=1)
            zero_count = np.sum(zero_vectors)
            
            if zero_count > 0:
                zero_indices = np.where(zero_vectors)[0].tolist()
                issues.append(ValidationIssue(
                    type=ValidationType.QUALITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Found {zero_count} zero vectors",
                    details={
                        "zero_vector_count": int(zero_count),
                        "affected_vector_indices": zero_indices[:50]
                    },
                    affected_items=[str(i) for i in zero_indices],
                    remediation_suggestion="Review source documents for zero vectors"
                ))
            else:
                passed_checks += 1
            
            # Check 4: Outlier detection (if enabled)
            if check_outliers:
                total_checks += 1
                outlier_issues = await self._detect_vector_outliers(vector_array)
                if outlier_issues:
                    issues.extend(outlier_issues)
                else:
                    passed_checks += 1
            
            # Statistical summary
            vector_norms = np.linalg.norm(vector_array, axis=1)
            vector_means = np.mean(vector_array, axis=1)
            
            quality_stats = {
                "norm_mean": float(np.mean(vector_norms)),
                "norm_std": float(np.std(vector_norms)),
                "norm_min": float(np.min(vector_norms)),
                "norm_max": float(np.max(vector_norms)),
                "mean_mean": float(np.mean(vector_means)),
                "mean_std": float(np.std(vector_means))
            }
        
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        report = ValidationReport(
            validation_id=validation_id,
            timestamp=datetime.now().isoformat(),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=total_checks - passed_checks,
            issues=issues,
            summary={
                "vector_count": len(vectors),
                "nan_vectors": int(nan_count),
                "inf_vectors": int(inf_count),
                "zero_vectors": int(zero_count),
                "quality_statistics": quality_stats
            },
            performance_metrics={
                "validation_time_seconds": elapsed_time,
                "vectors_per_second": len(vectors) / elapsed_time if elapsed_time > 0 else 0
            }
        )
        
        await self._update_validation_stats(report)
        
        self.logger.info(
            "embedding_quality_validation_completed",
            validation_id=validation_id,
            success_rate=report.success_rate,
            quality_issues=len(issues)
        )
        
        return report
    
    async def validate_id_uniqueness(
        self,
        document_ids: List[str],
        vector_ids: Optional[List[str]] = None
    ) -> ValidationReport:
        """
        Validate ID uniqueness across vector store operations.
        
        Args:
            document_ids: List of document IDs
            vector_ids: Optional list of vector IDs
            
        Returns:
            ValidationReport with uniqueness validation results
        """
        validation_id = f"uniqueness_{int(datetime.now().timestamp())}"
        start_time = asyncio.get_event_loop().time()
        
        self.logger.info(
            "id_uniqueness_validation_started",
            validation_id=validation_id,
            document_id_count=len(document_ids),
            has_vector_ids=vector_ids is not None
        )
        
        issues = []
        total_checks = 0
        passed_checks = 0
        
        async with performance_monitor.track_operation("uniqueness_validation", timeout=30.0):
            # Check 1: Document ID uniqueness
            total_checks += 1
            doc_id_counts = Counter(document_ids)
            duplicate_doc_ids = {id_val: count for id_val, count in doc_id_counts.items() if count > 1}
            
            if duplicate_doc_ids:
                issues.append(ValidationIssue(
                    type=ValidationType.UNIQUENESS,
                    severity=ValidationSeverity.ERROR,
                    message=f"Found {len(duplicate_doc_ids)} duplicate document IDs",
                    details={
                        "duplicate_ids": dict(list(duplicate_doc_ids.items())[:20]),  # Limit sample
                        "total_duplicates": len(duplicate_doc_ids),
                        "max_occurrences": max(duplicate_doc_ids.values())
                    },
                    affected_items=list(duplicate_doc_ids.keys()),
                    remediation_suggestion="Ensure document IDs are unique before vector storage"
                ))
            else:
                passed_checks += 1
            
            # Check 2: Vector ID uniqueness (if provided)
            if vector_ids is not None:
                total_checks += 1
                vector_id_counts = Counter(vector_ids)
                duplicate_vector_ids = {id_val: count for id_val, count in vector_id_counts.items() if count > 1}
                
                if duplicate_vector_ids:
                    issues.append(ValidationIssue(
                        type=ValidationType.UNIQUENESS,
                        severity=ValidationSeverity.ERROR,
                        message=f"Found {len(duplicate_vector_ids)} duplicate vector IDs",
                        details={
                            "duplicate_ids": dict(list(duplicate_vector_ids.items())[:20]),
                            "total_duplicates": len(duplicate_vector_ids),
                            "max_occurrences": max(duplicate_vector_ids.values())
                        },
                        affected_items=list(duplicate_vector_ids.keys()),
                        remediation_suggestion="Ensure vector IDs are unique"
                    ))
                else:
                    passed_checks += 1
                
                # Check 3: Document-Vector ID alignment
                if len(document_ids) == len(vector_ids):
                    total_checks += 1
                    mismatched_pairs = []
                    for i, (doc_id, vec_id) in enumerate(zip(document_ids, vector_ids)):
                        if doc_id != vec_id:
                            mismatched_pairs.append({"index": i, "doc_id": doc_id, "vec_id": vec_id})
                    
                    if mismatched_pairs:
                        issues.append(ValidationIssue(
                            type=ValidationType.UNIQUENESS,
                            severity=ValidationSeverity.WARNING,
                            message=f"Found {len(mismatched_pairs)} document-vector ID mismatches",
                            details={
                                "mismatch_count": len(mismatched_pairs),
                                "sample_mismatches": mismatched_pairs[:10]
                            },
                            remediation_suggestion="Review ID assignment strategy"
                        ))
                    else:
                        passed_checks += 1
        
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        report = ValidationReport(
            validation_id=validation_id,
            timestamp=datetime.now().isoformat(),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=total_checks - passed_checks,
            issues=issues,
            summary={
                "document_id_count": len(document_ids),
                "unique_document_ids": len(set(document_ids)),
                "duplicate_document_ids": len(duplicate_doc_ids),
                "vector_id_count": len(vector_ids) if vector_ids else 0,
                "unique_vector_ids": len(set(vector_ids)) if vector_ids else 0
            },
            performance_metrics={
                "validation_time_seconds": elapsed_time
            }
        )
        
        await self._update_validation_stats(report)
        
        self.logger.info(
            "id_uniqueness_validation_completed",
            validation_id=validation_id,
            success_rate=report.success_rate,
            duplicate_issues=len([i for i in issues if i.type == ValidationType.UNIQUENESS])
        )
        
        return report
    
    async def _validate_metadata_structure(
        self,
        metadatas: List[Dict[str, Any]]
    ) -> List[ValidationIssue]:
        """Validate metadata structure consistency."""
        if not metadatas:
            return []
        
        issues = []
        
        # Check for consistent keys across all metadata entries
        all_keys = set()
        key_frequencies = defaultdict(int)
        
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                issues.append(ValidationIssue(
                    type=ValidationType.ALIGNMENT,
                    severity=ValidationSeverity.ERROR,
                    message="Non-dictionary metadata entry found",
                    details={"type": str(type(metadata))},
                    remediation_suggestion="Ensure all metadata entries are dictionaries"
                ))
                continue
            
            for key in metadata.keys():
                all_keys.add(key)
                key_frequencies[key] += 1
        
        # Check for missing keys in some entries
        total_entries = len(metadatas)
        inconsistent_keys = []
        
        for key, frequency in key_frequencies.items():
            if frequency < total_entries * 0.95:  # Allow 5% tolerance
                inconsistent_keys.append({
                    "key": key,
                    "frequency": frequency,
                    "percentage": (frequency / total_entries) * 100
                })
        
        if inconsistent_keys:
            issues.append(ValidationIssue(
                type=ValidationType.ALIGNMENT,
                severity=ValidationSeverity.WARNING,
                message=f"Inconsistent metadata structure: {len(inconsistent_keys)} keys not present in all entries",
                details={
                    "inconsistent_keys": inconsistent_keys,
                    "total_entries": total_entries,
                    "unique_keys": len(all_keys)
                },
                remediation_suggestion="Standardize metadata structure across all entries"
            ))
        
        return issues
    
    async def _validate_field_alignment(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> List[ValidationIssue]:
        """Validate field alignment between vectors and metadata."""
        issues = []
        
        # Check if metadata contains vector-related fields that should align
        vector_related_fields = ["dimension", "vector_id", "embedding_model"]
        
        for i, (vector, metadata) in enumerate(zip(vectors, metadatas)):
            if "dimension" in metadata:
                expected_dim = metadata["dimension"]
                actual_dim = len(vector)
                if expected_dim != actual_dim:
                    issues.append(ValidationIssue(
                        type=ValidationType.ALIGNMENT,
                        severity=ValidationSeverity.ERROR,
                        message=f"Dimension mismatch at index {i}",
                        details={
                            "index": i,
                            "metadata_dimension": expected_dim,
                            "actual_dimension": actual_dim
                        },
                        affected_items=[str(i)],
                        remediation_suggestion="Update metadata dimension to match vector"
                    ))
        
        return issues
    
    async def _detect_vector_outliers(
        self,
        vector_array: np.ndarray,
        z_threshold: float = 3.0
    ) -> List[ValidationIssue]:
        """Detect outlier vectors using statistical analysis."""
        issues = []
        
        try:
            # Calculate vector norms
            norms = np.linalg.norm(vector_array, axis=1)
            
            # Z-score based outlier detection
            norm_mean = np.mean(norms)
            norm_std = np.std(norms)
            
            if norm_std > 0:
                z_scores = np.abs((norms - norm_mean) / norm_std)
                outlier_mask = z_scores > z_threshold
                outlier_indices = np.where(outlier_mask)[0]
                
                if len(outlier_indices) > 0:
                    outlier_count = len(outlier_indices)
                    outlier_percentage = (outlier_count / len(vector_array)) * 100
                    
                    severity = ValidationSeverity.WARNING
                    if outlier_percentage > 10:  # More than 10% outliers
                        severity = ValidationSeverity.ERROR
                    
                    issues.append(ValidationIssue(
                        type=ValidationType.QUALITY,
                        severity=severity,
                        message=f"Detected {outlier_count} statistical outliers ({outlier_percentage:.1f}%)",
                        details={
                            "outlier_count": int(outlier_count),
                            "outlier_percentage": outlier_percentage,
                            "z_threshold": z_threshold,
                            "norm_statistics": {
                                "mean": float(norm_mean),
                                "std": float(norm_std),
                                "min": float(np.min(norms)),
                                "max": float(np.max(norms))
                            },
                            "sample_outlier_indices": outlier_indices[:20].tolist()
                        },
                        affected_items=[str(i) for i in outlier_indices],
                        remediation_suggestion="Review outlier vectors for quality issues"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                type=ValidationType.QUALITY,
                severity=ValidationSeverity.WARNING,
                message=f"Outlier detection failed: {str(e)}",
                details={"error": str(e)},
                remediation_suggestion="Check vector data format"
            ))
        
        return issues
    
    async def _update_validation_stats(self, report: ValidationReport) -> None:
        """Update internal validation statistics."""
        self._validation_stats["total_validations"] += 1
        
        if report.failed_checks == 0:
            self._validation_stats["successful_validations"] += 1
        else:
            self._validation_stats["failed_validations"] += 1
        
        if report.has_critical_issues:
            self._validation_stats["critical_issues_found"] += 1
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self._validation_stats["total_validations"]
        success_rate = 0.0
        if total > 0:
            success_rate = (self._validation_stats["successful_validations"] / total) * 100.0
        
        return {
            **self._validation_stats,
            "success_rate_percentage": success_rate
        }


# Global instance for shared use
consistency_validator = VectorConsistencyValidator()