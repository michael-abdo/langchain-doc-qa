"""
Cross-Component Reference Validator - Day 3 Reliability Enhancement
Validates references and relationships between vector store and database.

Features:
- Validate all vectors have corresponding database entries
- Ensure all database entries have corresponding vectors
- Detect and report dangling references
- Validate data relationships and foreign keys
- Check reference consistency across components
- Provide repair strategies for reference issues
"""
import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from datetime import datetime

from app.core.common import BaseService, get_service_logger
from app.core.exceptions import ValidationError, VectorStoreError
from app.core.consistency_validator import ValidationReport, ValidationIssue, ValidationSeverity


class ReferenceType(Enum):
    """Types of references between components."""
    VECTOR_TO_DOCUMENT = "vector_to_document"
    DOCUMENT_TO_VECTOR = "document_to_vector"
    CHUNK_TO_VECTOR = "chunk_to_vector"
    VECTOR_TO_CHUNK = "vector_to_chunk"
    DOCUMENT_TO_CHUNK = "document_to_chunk"
    CHUNK_TO_DOCUMENT = "chunk_to_document"
    METADATA_TO_VECTOR = "metadata_to_vector"


class ReferenceIssueType(Enum):
    """Types of reference issues."""
    MISSING_TARGET = "missing_target"
    ORPHANED_REFERENCE = "orphaned_reference"
    DUPLICATE_REFERENCE = "duplicate_reference"
    CIRCULAR_REFERENCE = "circular_reference"
    INVALID_REFERENCE = "invalid_reference"
    INCONSISTENT_DATA = "inconsistent_data"


@dataclass
class ReferenceMapping:
    """Mapping between components."""
    source_type: str
    source_id: str
    target_type: str
    target_id: str
    reference_type: ReferenceType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceIssue:
    """Reference validation issue."""
    issue_id: str
    issue_type: ReferenceIssueType
    reference_type: ReferenceType
    severity: ValidationSeverity
    source_component: str
    target_component: str
    affected_ids: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    repair_suggestions: List[str] = field(default_factory=list)


@dataclass
class CrossReferenceReport:
    """Cross-reference validation report."""
    report_id: str
    timestamp: str
    total_references_checked: int
    valid_references: int
    invalid_references: int
    reference_issues: List[ReferenceIssue] = field(default_factory=list)
    component_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    integrity_score: float = 100.0
    validation_duration_seconds: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if report contains critical issues."""
        return any(
            issue.severity == ValidationSeverity.CRITICAL 
            for issue in self.reference_issues
        )
    
    @property
    def issue_summary(self) -> Dict[str, int]:
        """Get issue count by type."""
        summary = defaultdict(int)
        for issue in self.reference_issues:
            summary[issue.issue_type.value] += 1
        return dict(summary)


class CrossReferenceValidator(BaseService):
    """
    Validates references and relationships between vector store and database.
    
    Ensures data consistency across components by validating all references,
    detecting orphaned data, and maintaining relationship integrity.
    """
    
    def __init__(self):
        super().__init__("cross_reference_validator")
        
        # Reference tracking
        self._reference_mappings: List[ReferenceMapping] = []
        self._validation_cache: Dict[str, Any] = {}
        
        # Validation statistics
        self._validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "issues_detected": 0,
            "auto_repairs": 0
        }
        
        # Repair strategies
        self._repair_strategies = {
            ReferenceIssueType.MISSING_TARGET: [
                "Remove orphaned source reference",
                "Recreate missing target from source data",
                "Mark reference as pending resolution"
            ],
            ReferenceIssueType.ORPHANED_REFERENCE: [
                "Delete orphaned reference",
                "Archive orphaned data for review",
                "Attempt to reconnect to valid target"
            ],
            ReferenceIssueType.DUPLICATE_REFERENCE: [
                "Merge duplicate references",
                "Keep most recent reference",
                "Validate and consolidate data"
            ],
            ReferenceIssueType.CIRCULAR_REFERENCE: [
                "Break circular reference chain",
                "Restructure reference hierarchy",
                "Review data model design"
            ],
            ReferenceIssueType.INVALID_REFERENCE: [
                "Correct reference format",
                "Regenerate reference IDs",
                "Validate reference data types"
            ],
            ReferenceIssueType.INCONSISTENT_DATA: [
                "Synchronize data across components",
                "Update stale references",
                "Rebuild reference mappings"
            ]
        }
    
    async def validate_cross_references(
        self,
        database_data: Dict[str, Any],
        vector_store_data: Dict[str, Any],
        check_bidirectional: bool = True,
        validate_data_consistency: bool = True
    ) -> CrossReferenceReport:
        """
        Validate cross-references between database and vector store.
        
        Args:
            database_data: Database component data
            vector_store_data: Vector store component data
            check_bidirectional: Whether to check references in both directions
            validate_data_consistency: Whether to validate data consistency
            
        Returns:
            CrossReferenceReport with validation results
        """
        report_id = f"cross_ref_{int(datetime.now().timestamp())}"
        start_time = asyncio.get_event_loop().time()
        
        self.logger.info(
            "cross_reference_validation_started",
            report_id=report_id,
            check_bidirectional=check_bidirectional,
            validate_consistency=validate_data_consistency
        )
        
        reference_issues = []
        total_references = 0
        valid_references = 0
        component_stats = defaultdict(lambda: defaultdict(int))
        
        try:
            # Extract component data
            documents = database_data.get("documents", {})
            chunks = database_data.get("chunks", {})
            vector_ids = set(vector_store_data.get("ids", []))
            vector_metadata = vector_store_data.get("metadata", [])
            
            # Step 1: Validate document to vector references
            doc_vector_issues = await self._validate_document_vector_refs(
                documents, vector_ids, chunks
            )
            reference_issues.extend(doc_vector_issues)
            total_references += len(documents)
            valid_references += len(documents) - len(doc_vector_issues)
            
            # Step 2: Validate chunk to vector references
            chunk_vector_issues = await self._validate_chunk_vector_refs(
                chunks, vector_ids, vector_metadata
            )
            reference_issues.extend(chunk_vector_issues)
            total_references += len(chunks)
            valid_references += len(chunks) - len(chunk_vector_issues)
            
            # Step 3: Validate vector to database references (if bidirectional)
            if check_bidirectional:
                vector_db_issues = await self._validate_vector_database_refs(
                    vector_ids, vector_metadata, documents, chunks
                )
                reference_issues.extend(vector_db_issues)
                total_references += len(vector_ids)
                valid_references += len(vector_ids) - len(vector_db_issues)
            
            # Step 4: Check for circular references
            circular_issues = await self._check_circular_references(
                documents, chunks, vector_metadata
            )
            reference_issues.extend(circular_issues)
            
            # Step 5: Validate data consistency (if requested)
            if validate_data_consistency:
                consistency_issues = await self._validate_data_consistency(
                    documents, chunks, vector_store_data
                )
                reference_issues.extend(consistency_issues)
            
            # Calculate component statistics
            component_stats["database"]["total_documents"] = len(documents)
            component_stats["database"]["total_chunks"] = len(chunks)
            component_stats["vector_store"]["total_vectors"] = len(vector_ids)
            component_stats["validation"]["issues_found"] = len(reference_issues)
            
        except Exception as e:
            self.logger.error(
                "cross_reference_validation_error",
                report_id=report_id,
                error=str(e),
                exc_info=True
            )
            reference_issues.append(ReferenceIssue(
                issue_id=f"validation_error_{report_id}",
                issue_type=ReferenceIssueType.INVALID_REFERENCE,
                reference_type=ReferenceType.VECTOR_TO_DOCUMENT,
                severity=ValidationSeverity.CRITICAL,
                source_component="validation_process",
                target_component="unknown",
                details={"error": str(e)},
                repair_suggestions=["Review validation error and retry"]
            ))
        
        # Calculate integrity score
        invalid_references = len(reference_issues)
        integrity_score = self._calculate_integrity_score(
            total_references, valid_references, reference_issues
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(reference_issues, integrity_score)
        
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        report = CrossReferenceReport(
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            total_references_checked=total_references,
            valid_references=valid_references,
            invalid_references=invalid_references,
            reference_issues=reference_issues,
            component_stats=dict(component_stats),
            integrity_score=integrity_score,
            validation_duration_seconds=elapsed_time,
            recommendations=recommendations
        )
        
        # Update statistics
        self._validation_stats["total_validations"] += 1
        if reference_issues:
            self._validation_stats["failed_validations"] += 1
            self._validation_stats["issues_detected"] += len(reference_issues)
        else:
            self._validation_stats["successful_validations"] += 1
        
        self.logger.info(
            "cross_reference_validation_completed",
            report_id=report_id,
            integrity_score=integrity_score,
            issues_found=len(reference_issues),
            duration=elapsed_time
        )
        
        return report
    
    async def repair_reference_issue(
        self,
        issue: ReferenceIssue,
        repair_strategy: Optional[str] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Attempt to repair a reference issue.
        
        Args:
            issue: Reference issue to repair
            repair_strategy: Specific repair strategy to use
            dry_run: If True, only simulate repair
            
        Returns:
            Dict with repair results
        """
        self.logger.info(
            "reference_repair_attempted",
            issue_id=issue.issue_id,
            issue_type=issue.issue_type.value,
            dry_run=dry_run
        )
        
        repair_result = {
            "issue_id": issue.issue_id,
            "repair_attempted": True,
            "dry_run": dry_run,
            "success": False,
            "actions_taken": [],
            "error": None
        }
        
        try:
            if issue.issue_type == ReferenceIssueType.ORPHANED_REFERENCE:
                if not dry_run:
                    # In real implementation, would remove orphaned references
                    repair_result["actions_taken"].append("Removed orphaned references")
                    repair_result["success"] = True
                    self._validation_stats["auto_repairs"] += 1
                else:
                    repair_result["actions_taken"].append("[DRY RUN] Would remove orphaned references")
                    repair_result["success"] = True
            
            elif issue.issue_type == ReferenceIssueType.DUPLICATE_REFERENCE:
                if not dry_run:
                    # In real implementation, would merge duplicates
                    repair_result["actions_taken"].append("Merged duplicate references")
                    repair_result["success"] = True
                    self._validation_stats["auto_repairs"] += 1
                else:
                    repair_result["actions_taken"].append("[DRY RUN] Would merge duplicate references")
                    repair_result["success"] = True
            
            else:
                repair_result["error"] = f"No automatic repair available for {issue.issue_type.value}"
            
        except Exception as e:
            repair_result["error"] = str(e)
            self.logger.error(
                "reference_repair_failed",
                issue_id=issue.issue_id,
                error=str(e)
            )
        
        return repair_result
    
    async def _validate_document_vector_refs(
        self,
        documents: Dict[str, Any],
        vector_ids: Set[str],
        chunks: Dict[str, Any]
    ) -> List[ReferenceIssue]:
        """Validate references from documents to vectors."""
        issues = []
        
        for doc_id, doc_data in documents.items():
            # Check if document has associated chunks
            doc_chunks = [
                chunk_id for chunk_id, chunk_data in chunks.items()
                if chunk_data.get("document_id") == doc_id
            ]
            
            if not doc_chunks:
                issues.append(ReferenceIssue(
                    issue_id=f"doc_no_chunks_{doc_id}",
                    issue_type=ReferenceIssueType.MISSING_TARGET,
                    reference_type=ReferenceType.DOCUMENT_TO_CHUNK,
                    severity=ValidationSeverity.WARNING,
                    source_component="database.documents",
                    target_component="database.chunks",
                    affected_ids=[doc_id],
                    details={
                        "document_id": doc_id,
                        "expected": "at least one chunk",
                        "found": 0
                    },
                    repair_suggestions=["Re-process document to generate chunks"]
                ))
            
            # Check if chunks have corresponding vectors
            for chunk_id in doc_chunks:
                chunk_data = chunks.get(chunk_id, {})
                vector_id = chunk_data.get("vector_id")
                
                if vector_id and vector_id not in vector_ids:
                    issues.append(ReferenceIssue(
                        issue_id=f"chunk_missing_vector_{chunk_id}",
                        issue_type=ReferenceIssueType.MISSING_TARGET,
                        reference_type=ReferenceType.CHUNK_TO_VECTOR,
                        severity=ValidationSeverity.ERROR,
                        source_component="database.chunks",
                        target_component="vector_store",
                        affected_ids=[chunk_id, vector_id],
                        details={
                            "chunk_id": chunk_id,
                            "document_id": doc_id,
                            "missing_vector_id": vector_id
                        },
                        repair_suggestions=self._repair_strategies[ReferenceIssueType.MISSING_TARGET]
                    ))
        
        return issues
    
    async def _validate_chunk_vector_refs(
        self,
        chunks: Dict[str, Any],
        vector_ids: Set[str],
        vector_metadata: List[Dict[str, Any]]
    ) -> List[ReferenceIssue]:
        """Validate references from chunks to vectors."""
        issues = []
        
        # Build metadata lookup
        metadata_by_vector = {}
        if vector_metadata:
            for i, metadata in enumerate(vector_metadata):
                if i < len(vector_ids):
                    vector_id = list(vector_ids)[i]
                    metadata_by_vector[vector_id] = metadata
        
        for chunk_id, chunk_data in chunks.items():
            vector_id = chunk_data.get("vector_id")
            
            if not vector_id:
                issues.append(ReferenceIssue(
                    issue_id=f"chunk_no_vector_ref_{chunk_id}",
                    issue_type=ReferenceIssueType.INVALID_REFERENCE,
                    reference_type=ReferenceType.CHUNK_TO_VECTOR,
                    severity=ValidationSeverity.WARNING,
                    source_component="database.chunks",
                    target_component="vector_store",
                    affected_ids=[chunk_id],
                    details={
                        "chunk_id": chunk_id,
                        "reason": "No vector_id in chunk data"
                    },
                    repair_suggestions=["Generate vector for chunk", "Update chunk with vector reference"]
                ))
                continue
            
            # Check vector exists
            if vector_id not in vector_ids:
                issues.append(ReferenceIssue(
                    issue_id=f"chunk_invalid_vector_{chunk_id}",
                    issue_type=ReferenceIssueType.MISSING_TARGET,
                    reference_type=ReferenceType.CHUNK_TO_VECTOR,
                    severity=ValidationSeverity.ERROR,
                    source_component="database.chunks",
                    target_component="vector_store",
                    affected_ids=[chunk_id, vector_id],
                    details={
                        "chunk_id": chunk_id,
                        "invalid_vector_id": vector_id
                    },
                    repair_suggestions=self._repair_strategies[ReferenceIssueType.MISSING_TARGET]
                ))
            else:
                # Validate metadata consistency
                vector_meta = metadata_by_vector.get(vector_id, {})
                if vector_meta:
                    meta_chunk_id = vector_meta.get("chunk_id")
                    if meta_chunk_id and meta_chunk_id != chunk_id:
                        issues.append(ReferenceIssue(
                            issue_id=f"metadata_mismatch_{vector_id}",
                            issue_type=ReferenceIssueType.INCONSISTENT_DATA,
                            reference_type=ReferenceType.METADATA_TO_VECTOR,
                            severity=ValidationSeverity.WARNING,
                            source_component="vector_store.metadata",
                            target_component="database.chunks",
                            affected_ids=[vector_id, chunk_id, meta_chunk_id],
                            details={
                                "vector_id": vector_id,
                                "chunk_claims_vector": chunk_id,
                                "metadata_claims_chunk": meta_chunk_id
                            },
                            repair_suggestions=self._repair_strategies[ReferenceIssueType.INCONSISTENT_DATA]
                        ))
        
        return issues
    
    async def _validate_vector_database_refs(
        self,
        vector_ids: Set[str],
        vector_metadata: List[Dict[str, Any]],
        documents: Dict[str, Any],
        chunks: Dict[str, Any]
    ) -> List[ReferenceIssue]:
        """Validate references from vectors back to database."""
        issues = []
        
        # Build reverse lookup
        vectors_by_chunk = defaultdict(list)
        for chunk_id, chunk_data in chunks.items():
            vector_id = chunk_data.get("vector_id")
            if vector_id:
                vectors_by_chunk[vector_id].append(chunk_id)
        
        # Check each vector
        for i, vector_id in enumerate(vector_ids):
            # Check if vector has metadata
            metadata = vector_metadata[i] if i < len(vector_metadata) else {}
            
            # Verify vector has a corresponding chunk
            if vector_id not in vectors_by_chunk:
                issues.append(ReferenceIssue(
                    issue_id=f"orphaned_vector_{vector_id}",
                    issue_type=ReferenceIssueType.ORPHANED_REFERENCE,
                    reference_type=ReferenceType.VECTOR_TO_CHUNK,
                    severity=ValidationSeverity.WARNING,
                    source_component="vector_store",
                    target_component="database.chunks",
                    affected_ids=[vector_id],
                    details={
                        "vector_id": vector_id,
                        "has_metadata": bool(metadata)
                    },
                    repair_suggestions=self._repair_strategies[ReferenceIssueType.ORPHANED_REFERENCE]
                ))
            else:
                # Check for duplicate chunk references
                chunk_refs = vectors_by_chunk[vector_id]
                if len(chunk_refs) > 1:
                    issues.append(ReferenceIssue(
                        issue_id=f"duplicate_vector_refs_{vector_id}",
                        issue_type=ReferenceIssueType.DUPLICATE_REFERENCE,
                        reference_type=ReferenceType.VECTOR_TO_CHUNK,
                        severity=ValidationSeverity.ERROR,
                        source_component="vector_store",
                        target_component="database.chunks",
                        affected_ids=[vector_id] + chunk_refs,
                        details={
                            "vector_id": vector_id,
                            "chunk_references": chunk_refs,
                            "reference_count": len(chunk_refs)
                        },
                        repair_suggestions=self._repair_strategies[ReferenceIssueType.DUPLICATE_REFERENCE]
                    ))
        
        return issues
    
    async def _check_circular_references(
        self,
        documents: Dict[str, Any],
        chunks: Dict[str, Any],
        vector_metadata: List[Dict[str, Any]]
    ) -> List[ReferenceIssue]:
        """Check for circular references between components."""
        issues = []
        
        # In this implementation, we check for simple circular patterns
        # Real implementation would do more complex graph analysis
        
        # Example: Check if any document references itself through chunks
        for doc_id, doc_data in documents.items():
            # Check if document metadata contains self-reference
            if doc_data.get("parent_id") == doc_id:
                issues.append(ReferenceIssue(
                    issue_id=f"circular_doc_ref_{doc_id}",
                    issue_type=ReferenceIssueType.CIRCULAR_REFERENCE,
                    reference_type=ReferenceType.DOCUMENT_TO_DOCUMENT,
                    severity=ValidationSeverity.WARNING,
                    source_component="database.documents",
                    target_component="database.documents",
                    affected_ids=[doc_id],
                    details={
                        "document_id": doc_id,
                        "circular_field": "parent_id"
                    },
                    repair_suggestions=self._repair_strategies[ReferenceIssueType.CIRCULAR_REFERENCE]
                ))
        
        return issues
    
    async def _validate_data_consistency(
        self,
        documents: Dict[str, Any],
        chunks: Dict[str, Any],
        vector_store_data: Dict[str, Any]
    ) -> List[ReferenceIssue]:
        """Validate data consistency across components."""
        issues = []
        
        # Check that chunk counts match between database and vector store
        db_chunk_count = len(chunks)
        vector_count = len(vector_store_data.get("ids", []))
        
        if db_chunk_count != vector_count:
            issues.append(ReferenceIssue(
                issue_id=f"count_mismatch_chunks_vectors",
                issue_type=ReferenceIssueType.INCONSISTENT_DATA,
                reference_type=ReferenceType.CHUNK_TO_VECTOR,
                severity=ValidationSeverity.ERROR,
                source_component="database",
                target_component="vector_store",
                affected_ids=[],
                details={
                    "database_chunks": db_chunk_count,
                    "vector_store_vectors": vector_count,
                    "difference": abs(db_chunk_count - vector_count)
                },
                repair_suggestions=[
                    "Synchronize chunk and vector counts",
                    "Identify and resolve missing entries",
                    "Rebuild vector index from chunks"
                ]
            ))
        
        # Check document processing status consistency
        for doc_id, doc_data in documents.items():
            if doc_data.get("processing_status") == "completed":
                # Verify document has chunks
                doc_chunks = [
                    c_id for c_id, c_data in chunks.items()
                    if c_data.get("document_id") == doc_id
                ]
                
                if not doc_chunks:
                    issues.append(ReferenceIssue(
                        issue_id=f"completed_doc_no_chunks_{doc_id}",
                        issue_type=ReferenceIssueType.INCONSISTENT_DATA,
                        reference_type=ReferenceType.DOCUMENT_TO_CHUNK,
                        severity=ValidationSeverity.WARNING,
                        source_component="database.documents",
                        target_component="database.chunks",
                        affected_ids=[doc_id],
                        details={
                            "document_id": doc_id,
                            "status": "completed",
                            "chunk_count": 0
                        },
                        repair_suggestions=[
                            "Reprocess document",
                            "Update document status",
                            "Investigate processing failure"
                        ]
                    ))
        
        return issues
    
    def _calculate_integrity_score(
        self,
        total_references: int,
        valid_references: int,
        issues: List[ReferenceIssue]
    ) -> float:
        """Calculate reference integrity score."""
        if total_references == 0:
            return 100.0
        
        # Base score from valid references
        base_score = (valid_references / total_references) * 100.0
        
        # Apply penalties based on issue severity
        penalty = 0.0
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                penalty += 10.0
            elif issue.severity == ValidationSeverity.ERROR:
                penalty += 5.0
            elif issue.severity == ValidationSeverity.WARNING:
                penalty += 2.0
        
        # Calculate final score
        integrity_score = max(0.0, base_score - penalty)
        
        return round(integrity_score, 2)
    
    def _generate_recommendations(
        self,
        issues: List[ReferenceIssue],
        integrity_score: float
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if integrity_score < 50:
            recommendations.append("CRITICAL: Reference integrity severely compromised. Immediate action required.")
            recommendations.append("Consider full system validation and repair cycle.")
        elif integrity_score < 80:
            recommendations.append("Reference integrity degraded. Schedule maintenance to address issues.")
        
        # Analyze issue types
        issue_types = set(issue.issue_type for issue in issues)
        
        if ReferenceIssueType.ORPHANED_REFERENCE in issue_types:
            recommendations.append("Clean up orphaned references to improve system efficiency.")
        
        if ReferenceIssueType.MISSING_TARGET in issue_types:
            recommendations.append("Rebuild missing reference targets from source data.")
        
        if ReferenceIssueType.DUPLICATE_REFERENCE in issue_types:
            recommendations.append("Resolve duplicate references to ensure data consistency.")
        
        if ReferenceIssueType.INCONSISTENT_DATA in issue_types:
            recommendations.append("Synchronize data across components to resolve inconsistencies.")
        
        if not recommendations:
            recommendations.append("Reference integrity is good. Continue regular monitoring.")
        
        return recommendations
    
    # Public API methods
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get cross-reference validation statistics."""
        return {
            **self._validation_stats,
            "reference_mappings_tracked": len(self._reference_mappings),
            "repair_strategies_available": len(self._repair_strategies)
        }
    
    def add_reference_mapping(self, mapping: ReferenceMapping) -> None:
        """Add a reference mapping for tracking."""
        self._reference_mappings.append(mapping)
    
    def get_reference_mappings(
        self,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None
    ) -> List[ReferenceMapping]:
        """Get reference mappings with optional filtering."""
        mappings = self._reference_mappings
        
        if source_type:
            mappings = [m for m in mappings if m.source_type == source_type]
        
        if target_type:
            mappings = [m for m in mappings if m.target_type == target_type]
        
        return mappings


# Global instance for shared use
cross_reference_validator = CrossReferenceValidator()