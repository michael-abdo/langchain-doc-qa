"""
Vector Store Backup Manager - Day 3 Reliability Enhancement
Provides comprehensive backup and recovery capabilities for vector indices with integrity validation.

Features:
- Compressed backup file format with metadata headers
- Incremental and full backup mechanisms
- Integrity validation with checksums
- Automated scheduling and cleanup policies
- Cross-platform compatibility
"""
import asyncio
import gzip
import json
import hashlib
import struct
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, BinaryIO
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import shutil
import sqlite3
import pickle
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    # Graceful fallback if apscheduler is not installed
    AsyncIOScheduler = None
    CronTrigger = None
    IntervalTrigger = None
    SCHEDULER_AVAILABLE = False

from app.core.common import get_service_logger, BaseService, datetime as dt
from app.core.exceptions import VectorStoreError, ValidationError
from app.core.vector_reliability import memory_manager, performance_monitor


class BackupType(Enum):
    """Backup operation types."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class CompressionType(Enum):
    """Supported compression algorithms."""
    GZIP = "gzip"
    NONE = "none"


@dataclass
class BackupMetadata:
    """Metadata structure for backup files."""
    backup_id: str
    backup_type: BackupType
    created_at: str
    source_path: str
    original_size_bytes: int
    compressed_size_bytes: int
    compression_type: CompressionType
    file_count: int
    vector_count: int
    dimension: int
    schema_version: str
    checksum_algorithm: str
    data_checksum: str
    metadata_checksum: str
    parent_backup_id: Optional[str] = None
    retention_policy: Optional[Dict[str, Any]] = None
    custom_metadata: Optional[Dict[str, Any]] = None


@dataclass
class BackupFileHeader:
    """Binary file header structure for backup files."""
    magic_number: bytes = b'VSBK'  # Vector Store Backup
    version: int = 1
    metadata_size: int = 0
    metadata_offset: int = 16  # After header
    data_offset: int = 0
    flags: int = 0  # Reserved for future use


@dataclass
class BackupSchedule:
    """Backup scheduling configuration."""
    schedule_id: str
    source_path: str
    backup_type: BackupType
    cron_expression: str
    enabled: bool = True
    retention_days: int = 30
    max_backups: int = 100
    compression: CompressionType = CompressionType.GZIP
    custom_metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetentionPolicy:
    """Backup retention policy configuration."""
    max_age_days: int = 30
    max_backup_count: int = 100
    keep_daily: int = 7   # Keep daily backups for 7 days
    keep_weekly: int = 4  # Keep weekly backups for 4 weeks
    keep_monthly: int = 12  # Keep monthly backups for 12 months
    min_free_space_gb: float = 10.0  # Minimum free space in GB


class VectorBackupManager(BaseService):
    """
    Manages vector store backup and recovery operations.
    
    File Format:
    [Header: 16 bytes] [Metadata: JSON] [Data: Compressed Vector Index Files]
    """
    
    def __init__(self, backup_root: str = "/tmp/vector_backups"):
        super().__init__("vector_backup_manager")
        
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.default_compression = CompressionType.GZIP
        self.default_retention_days = 30
        self.max_concurrent_backups = 2
        self.backup_chunk_size = 8192
        
        # State tracking
        self._active_backups: Dict[str, Dict[str, Any]] = {}
        self._backup_history: List[BackupMetadata] = []
        self._last_cleanup = None
        
        # Incremental backup tracking
        self._change_tracker_db = self.backup_root / "change_tracker.db"
        self._initialize_change_tracker()
        
        # Automated scheduling
        self._scheduler = AsyncIOScheduler() if SCHEDULER_AVAILABLE else None
        self._schedules: Dict[str, BackupSchedule] = {}
        self._retention_policy = RetentionPolicy()
        self._scheduler_running = False
        
        self.logger.info(
            "vector_backup_manager_initialized",
            backup_root=str(self.backup_root),
            compression=self.default_compression.value,
            change_tracker=str(self._change_tracker_db)
        )
    
    async def create_backup(
        self,
        source_path: str,
        backup_type: BackupType = BackupType.FULL,
        compression: CompressionType = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        parent_backup_id: Optional[str] = None
    ) -> str:
        """
        Create a backup of vector store data.
        
        Args:
            source_path: Path to source vector index files
            backup_type: Type of backup to create
            compression: Compression algorithm to use
            custom_metadata: Additional metadata to include
            parent_backup_id: Parent backup for incremental backups
            
        Returns:
            Backup ID for the created backup
            
        Raises:
            VectorStoreError: If backup creation fails
        """
        # Memory check before backup
        estimated_memory_mb = self._estimate_backup_memory(source_path)
        if not memory_manager.check_memory_available(estimated_memory_mb):
            raise VectorStoreError(
                f"Insufficient memory for backup (requires ~{estimated_memory_mb:.1f}MB)",
                "memory_exhaustion"
            )
        
        backup_id = self._generate_backup_id()
        compression = compression or self.default_compression
        
        async with performance_monitor.track_operation("create_backup", timeout=1800.0):  # 30 min timeout
            try:
                self.logger.info(
                    "backup_creation_started",
                    backup_id=backup_id,
                    backup_type=backup_type.value,
                    source_path=source_path
                )
                
                # Track active backup
                self._active_backups[backup_id] = {
                    "started_at": datetime.now().isoformat(),
                    "backup_type": backup_type.value,
                    "source_path": source_path,
                    "status": "in_progress"
                }
                
                # Collect source files and metadata
                source_files = await self._collect_source_files(source_path)
                if not source_files:
                    raise VectorStoreError(
                        f"No files found at source path: {source_path}",
                        "invalid_source"
                    )
                
                # Extract vector store metadata
                vector_metadata = await self._extract_vector_metadata(source_files)
                
                # Create backup file path
                backup_filename = f"{backup_id}_{backup_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vsbk"
                backup_file_path = self.backup_root / backup_filename
                
                # Create backup metadata
                metadata = BackupMetadata(
                    backup_id=backup_id,
                    backup_type=backup_type,
                    created_at=datetime.now().isoformat(),
                    source_path=source_path,
                    original_size_bytes=sum(os.path.getsize(f) for f in source_files),
                    compressed_size_bytes=0,  # Will be set after compression
                    compression_type=compression,
                    file_count=len(source_files),
                    vector_count=vector_metadata.get("vector_count", 0),
                    dimension=vector_metadata.get("dimension", 0),
                    schema_version=vector_metadata.get("schema_version", "unknown"),
                    checksum_algorithm="sha256",
                    data_checksum="",  # Will be calculated
                    metadata_checksum="",  # Will be calculated
                    parent_backup_id=parent_backup_id,
                    custom_metadata=custom_metadata or {}
                )
                
                # For incremental backups, filter to only changed files
                if backup_type == BackupType.INCREMENTAL and parent_backup_id:
                    changed_files_info = await self._detect_changed_files(source_path, parent_backup_id)
                    if changed_files_info:
                        source_files = [cf["file_path"] for cf in changed_files_info]
                        metadata.custom_metadata["changed_files_info"] = changed_files_info
                
                # Write backup file
                await self._write_backup_file(backup_file_path, source_files, metadata, compression)
                
                # Update tracking
                self._active_backups[backup_id]["status"] = "completed"
                self._active_backups[backup_id]["backup_file"] = str(backup_file_path)
                self._backup_history.append(metadata)
                
                self.logger.info(
                    "backup_creation_completed",
                    backup_id=backup_id,
                    backup_file=str(backup_file_path),
                    original_size_mb=metadata.original_size_bytes / (1024*1024),
                    compressed_size_mb=metadata.compressed_size_bytes / (1024*1024),
                    compression_ratio=metadata.original_size_bytes / max(metadata.compressed_size_bytes, 1)
                )
                
                return backup_id
                
            except Exception as e:
                # Clean up failed backup
                if backup_id in self._active_backups:
                    self._active_backups[backup_id]["status"] = "failed"
                    self._active_backups[backup_id]["error"] = str(e)
                
                self.logger.error(
                    "backup_creation_failed",
                    backup_id=backup_id,
                    error=str(e)
                )
                raise VectorStoreError(f"Backup creation failed: {str(e)}", "backup_failed")
            
            finally:
                # Clean up tracking after completion or failure
                if backup_id in self._active_backups:
                    del self._active_backups[backup_id]
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"backup_{timestamp}_{random_suffix}"
    
    def _estimate_backup_memory(self, source_path: str) -> float:
        """Estimate memory requirements for backup operation."""
        try:
            source_dir = Path(source_path)
            total_size = sum(f.stat().st_size for f in source_dir.rglob('*') if f.is_file())
            # Estimate 2x source size for compression buffer
            return (total_size * 2) / (1024 * 1024)  # Convert to MB
        except Exception:
            return 100.0  # Default conservative estimate
    
    async def _collect_source_files(self, source_path: str) -> List[str]:
        """Collect all files to include in backup."""
        source_dir = Path(source_path)
        if not source_dir.exists():
            raise VectorStoreError(f"Source path does not exist: {source_path}", "invalid_source")
        
        # Include FAISS index files, metadata, and checksums
        file_patterns = ["*.faiss", "*.pkl", "*.checksum", "*.json"]
        source_files = []
        
        for pattern in file_patterns:
            source_files.extend(source_dir.glob(pattern))
        
        return [str(f) for f in source_files if f.is_file()]
    
    async def _extract_vector_metadata(self, source_files: List[str]) -> Dict[str, Any]:
        """Extract metadata about the vector store from source files."""
        metadata = {
            "vector_count": 0,
            "dimension": 0,
            "schema_version": "unknown"
        }
        
        # Try to read FAISS index to get vector count and dimension
        faiss_files = [f for f in source_files if f.endswith('.faiss')]
        if faiss_files:
            try:
                import faiss
                index = faiss.read_index(faiss_files[0])
                metadata["vector_count"] = index.ntotal
                metadata["dimension"] = index.d
            except Exception as e:
                self.logger.warning("failed_to_read_faiss_metadata", error=str(e))
        
        return metadata
    
    def _initialize_change_tracker(self) -> None:
        """Initialize SQLite database for tracking file changes."""
        try:
            conn = sqlite3.connect(str(self._change_tracker_db))
            cursor = conn.cursor()
            
            # Create table for tracking file states
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_changes (
                    file_path TEXT PRIMARY KEY,
                    last_modified REAL,
                    file_size INTEGER,
                    content_hash TEXT,
                    last_backup_id TEXT,
                    last_backup_timestamp TEXT,
                    change_type TEXT
                )
            """)
            
            # Create table for backup relationships
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backup_chain (
                    backup_id TEXT PRIMARY KEY,
                    parent_backup_id TEXT,
                    backup_type TEXT,
                    created_at TEXT,
                    file_count INTEGER
                )
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info("change_tracker_initialized", db_path=str(self._change_tracker_db))
            
        except Exception as e:
            self.logger.error("change_tracker_init_failed", error=str(e))
    
    async def create_incremental_backup(
        self,
        source_path: str,
        parent_backup_id: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create incremental backup with only changed files since parent backup."""
        # Find changed files since parent backup
        changed_files = await self._detect_changed_files(source_path, parent_backup_id)
        
        if not changed_files:
            self.logger.info(
                "no_changes_detected_for_incremental_backup",
                source_path=source_path,
                parent_backup_id=parent_backup_id
            )
            return None
        
        # Create incremental backup with only changed files
        return await self.create_backup(
            source_path,
            backup_type=BackupType.INCREMENTAL,
            custom_metadata={
                **(custom_metadata or {}),
                "changed_files": changed_files,
                "parent_backup_id": parent_backup_id
            },
            parent_backup_id=parent_backup_id
        )
    
    async def _detect_changed_files(self, source_path: str, parent_backup_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect files that have changed since the last backup."""
        try:
            source_files = await self._collect_source_files(source_path)
            changed_files = []
            
            conn = sqlite3.connect(str(self._change_tracker_db))
            cursor = conn.cursor()
            
            for file_path in source_files:
                file_stat = os.stat(file_path)
                current_mtime = file_stat.st_mtime
                current_size = file_stat.st_size
                
                # Calculate current file hash
                current_hash = await self._calculate_file_hash(file_path)
                
                # Check if file exists in tracking database
                cursor.execute(
                    "SELECT last_modified, file_size, content_hash, last_backup_id FROM file_changes WHERE file_path = ?",
                    (file_path,)
                )
                row = cursor.fetchone()
                
                change_type = "unknown"
                
                if row is None:
                    # New file
                    change_type = "added"
                    changed_files.append({
                        "file_path": file_path,
                        "change_type": change_type,
                        "current_hash": current_hash,
                        "size_bytes": current_size
                    })
                else:
                    last_modified, last_size, last_hash, last_backup_id = row
                    
                    # Check for changes
                    if (current_mtime > last_modified or 
                        current_size != last_size or 
                        current_hash != last_hash):
                        
                        if current_hash != last_hash:
                            change_type = "modified"
                        elif current_size != last_size:
                            change_type = "size_changed"
                        else:
                            change_type = "timestamp_changed"
                        
                        changed_files.append({
                            "file_path": file_path,
                            "change_type": change_type,
                            "current_hash": current_hash,
                            "previous_hash": last_hash,
                            "size_bytes": current_size,
                            "previous_size": last_size
                        })
            
            conn.close()
            
            self.logger.info(
                "change_detection_completed",
                total_files=len(source_files),
                changed_files=len(changed_files),
                parent_backup_id=parent_backup_id
            )
            
            return changed_files
            
        except Exception as e:
            self.logger.error("change_detection_failed", error=str(e))
            return []
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file content."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.warning("file_hash_calculation_failed", file_path=file_path, error=str(e))
            return "hash_error"
    
    async def _update_change_tracker(self, backup_id: str, source_files: List[str], backup_type: BackupType) -> None:
        """Update change tracker database after successful backup."""
        try:
            conn = sqlite3.connect(str(self._change_tracker_db))
            cursor = conn.cursor()
            
            current_timestamp = datetime.now().isoformat()
            
            # Update file tracking information
            for file_path in source_files:
                try:
                    file_stat = os.stat(file_path)
                    file_hash = await self._calculate_file_hash(file_path)
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO file_changes 
                        (file_path, last_modified, file_size, content_hash, last_backup_id, last_backup_timestamp, change_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        file_path,
                        file_stat.st_mtime,
                        file_stat.st_size,
                        file_hash,
                        backup_id,
                        current_timestamp,
                        "backed_up"
                    ))
                    
                except Exception as e:
                    self.logger.warning("file_tracking_update_failed", file_path=file_path, error=str(e))
            
            # Update backup chain
            cursor.execute("""
                INSERT OR REPLACE INTO backup_chain
                (backup_id, parent_backup_id, backup_type, created_at, file_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                backup_id,
                None,  # Will be set for incremental backups
                backup_type.value,
                current_timestamp,
                len(source_files)
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(
                "change_tracker_updated",
                backup_id=backup_id,
                tracked_files=len(source_files)
            )
            
        except Exception as e:
            self.logger.error("change_tracker_update_failed", backup_id=backup_id, error=str(e))
    
    async def get_backup_chain(self, backup_id: str) -> List[Dict[str, Any]]:
        """Get the full chain of backups for incremental restore."""
        try:
            conn = sqlite3.connect(str(self._change_tracker_db))
            cursor = conn.cursor()
            
            # Build backup chain from child to root
            chain = []
            current_id = backup_id
            
            while current_id:
                cursor.execute(
                    "SELECT backup_id, parent_backup_id, backup_type, created_at, file_count FROM backup_chain WHERE backup_id = ?",
                    (current_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    backup_info = {
                        "backup_id": row[0],
                        "parent_backup_id": row[1],
                        "backup_type": row[2],
                        "created_at": row[3],
                        "file_count": row[4]
                    }
                    chain.append(backup_info)
                    current_id = row[1]  # Move to parent
                else:
                    break
            
            conn.close()
            
            # Reverse to get root-to-child order
            chain.reverse()
            
            self.logger.info(
                "backup_chain_retrieved",
                backup_id=backup_id,
                chain_length=len(chain)
            )
            
            return chain
            
        except Exception as e:
            self.logger.error("backup_chain_retrieval_failed", backup_id=backup_id, error=str(e))
            return []
    
    async def _write_backup_file(self, backup_path: Path, source_files: List[str], metadata: BackupMetadata, compression: CompressionType) -> None:
        """Write backup file with header, metadata, and compressed data."""
        try:
            with open(backup_path, 'wb') as backup_file:
                # Write file header
                header = BackupFileHeader()
                
                # Create metadata JSON
                metadata_dict = asdict(metadata)
                metadata_json = json.dumps(metadata_dict, indent=2).encode('utf-8')
                header.metadata_size = len(metadata_json)
                header.data_offset = header.metadata_offset + header.metadata_size
                
                # Write binary header (16 bytes)
                backup_file.write(header.magic_number)  # 4 bytes
                backup_file.write(struct.pack('<I', header.version))  # 4 bytes
                backup_file.write(struct.pack('<I', header.metadata_size))  # 4 bytes
                backup_file.write(struct.pack('<I', header.data_offset))  # 4 bytes
                
                # Write metadata
                backup_file.write(metadata_json)
                
                # Write compressed data
                data_start_pos = backup_file.tell()
                data_hash = hashlib.sha256()
                
                if compression == CompressionType.GZIP:
                    # Use gzip compression
                    with gzip.open(backup_file, 'wb', compresslevel=6) as gzip_file:
                        await self._write_files_to_stream(gzip_file, source_files, data_hash)
                else:
                    # No compression
                    await self._write_files_to_stream(backup_file, source_files, data_hash)
                
                # Calculate final size and update metadata
                final_size = backup_file.tell()
                metadata.compressed_size_bytes = final_size - data_start_pos
                metadata.data_checksum = data_hash.hexdigest()
                
                # Calculate metadata checksum
                metadata_hash = hashlib.sha256(metadata_json).hexdigest()
                metadata.metadata_checksum = metadata_hash
                
                # Update metadata in file
                backup_file.seek(header.metadata_offset)
                updated_metadata_json = json.dumps(asdict(metadata), indent=2).encode('utf-8')
                backup_file.write(updated_metadata_json)
                
                # Pad to original size if needed
                if len(updated_metadata_json) < header.metadata_size:
                    padding = header.metadata_size - len(updated_metadata_json)
                    backup_file.write(b' ' * padding)
            
            # Update change tracker after successful backup
            await self._update_change_tracker(metadata.backup_id, source_files, metadata.backup_type)
            
            # Apply retention policy to metadata
            metadata.retention_policy = {
                "max_age_days": self._retention_policy.max_age_days,
                "created_by_schedule": metadata.custom_metadata.get("scheduled", False) if metadata.custom_metadata else False
            }
            
            self.logger.info(
                "backup_file_written",
                backup_path=str(backup_path),
                compressed_size_mb=metadata.compressed_size_bytes / (1024*1024),
                data_checksum=metadata.data_checksum[:16]
            )
            
        except Exception as e:
            self.logger.error("backup_file_write_failed", backup_path=str(backup_path), error=str(e))
            # Clean up partial file
            if backup_path.exists():
                backup_path.unlink()
            raise VectorStoreError(f"Failed to write backup file: {str(e)}", "backup_write_failed")
    
    async def _write_files_to_stream(self, stream: BinaryIO, source_files: List[str], data_hash: hashlib.sha256) -> None:
        """Write source files to backup stream with file metadata."""
        # Write file count
        file_count_bytes = struct.pack('<I', len(source_files))
        stream.write(file_count_bytes)
        data_hash.update(file_count_bytes)
        
        for file_path in source_files:
            try:
                file_path_obj = Path(file_path)
                
                # Write file metadata
                file_name = file_path_obj.name
                file_name_bytes = file_name.encode('utf-8')
                file_size = file_path_obj.stat().st_size
                
                # File metadata header: name_length(4) + name + file_size(8)
                name_length_bytes = struct.pack('<I', len(file_name_bytes))
                file_size_bytes = struct.pack('<Q', file_size)
                
                stream.write(name_length_bytes)
                stream.write(file_name_bytes)
                stream.write(file_size_bytes)
                
                data_hash.update(name_length_bytes)
                data_hash.update(file_name_bytes)
                data_hash.update(file_size_bytes)
                
                # Write file content
                with open(file_path, 'rb') as source_file:
                    while True:
                        chunk = source_file.read(self.backup_chunk_size)
                        if not chunk:
                            break
                        stream.write(chunk)
                        data_hash.update(chunk)
                        
            except Exception as e:
                self.logger.error("file_write_to_backup_failed", file_path=file_path, error=str(e))
                raise
    
    async def restore_backup(
        self,
        backup_file_path: str,
        restore_path: str,
        validate_integrity: bool = True,
        overwrite_existing: bool = False
    ) -> Dict[str, Any]:
        """Restore vector store data from backup with full integrity validation."""
        async with performance_monitor.track_operation("restore_backup", timeout=1800.0):
            try:
                self.logger.info(
                    "backup_restore_started",
                    backup_file=backup_file_path,
                    restore_path=restore_path
                )
                
                # Read and validate backup file
                metadata, data_stream_info = await self._read_backup_file(backup_file_path)
                
                if validate_integrity:
                    await self._validate_backup_integrity(backup_file_path, metadata)
                
                # Prepare restore directory
                restore_dir = Path(restore_path)
                if restore_dir.exists() and not overwrite_existing:
                    raise VectorStoreError(
                        f"Restore path already exists: {restore_path}",
                        "restore_path_exists"
                    )
                
                restore_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract files to restore directory
                extracted_files = await self._extract_backup_data(backup_file_path, restore_dir, metadata)
                
                restore_result = {
                    "backup_id": metadata.backup_id,
                    "backup_type": metadata.backup_type.value,
                    "restored_files": len(extracted_files),
                    "vector_count": metadata.vector_count,
                    "dimension": metadata.dimension,
                    "restore_path": str(restore_dir),
                    "integrity_validated": validate_integrity,
                    "extracted_files": extracted_files
                }
                
                self.logger.info(
                    "backup_restore_completed",
                    **{k: v for k, v in restore_result.items() if k != "extracted_files"}
                )
                
                return restore_result
                
            except Exception as e:
                self.logger.error(
                    "backup_restore_failed",
                    backup_file=backup_file_path,
                    error=str(e)
                )
                raise VectorStoreError(f"Backup restore failed: {str(e)}", "restore_failed")
    
    async def _read_backup_file(self, backup_path: str, read_data: bool = True) -> tuple:
        """Read backup file and return metadata and data stream info."""
        try:
            with open(backup_path, 'rb') as backup_file:
                # Read header
                magic = backup_file.read(4)
                if magic != b'VSBK':
                    raise VectorStoreError(f"Invalid backup file format: {backup_path}", "invalid_format")
                
                version = struct.unpack('<I', backup_file.read(4))[0]
                metadata_size = struct.unpack('<I', backup_file.read(4))[0]
                data_offset = struct.unpack('<I', backup_file.read(4))[0]
                
                # Read metadata
                metadata_json = backup_file.read(metadata_size).decode('utf-8').strip()
                metadata_dict = json.loads(metadata_json)
                
                # Convert back to BackupMetadata object
                metadata_dict['backup_type'] = BackupType(metadata_dict['backup_type'])
                metadata_dict['compression_type'] = CompressionType(metadata_dict['compression_type'])
                metadata = BackupMetadata(**metadata_dict)
                
                data_stream_info = {
                    "file_path": backup_path,
                    "data_offset": data_offset,
                    "compression": metadata.compression_type
                }
                
                return metadata, data_stream_info
                
        except Exception as e:
            self.logger.error("backup_file_read_failed", backup_path=backup_path, error=str(e))
            raise VectorStoreError(f"Failed to read backup file: {str(e)}", "backup_read_failed")
    
    async def _validate_backup_integrity(self, backup_path: str, metadata: BackupMetadata) -> bool:
        """Validate backup file integrity using checksums."""
        try:
            self.logger.info("backup_integrity_validation_started", backup_path=backup_path)
            
            # Validate metadata checksum
            with open(backup_path, 'rb') as backup_file:
                # Skip header and read metadata
                backup_file.seek(16)  # Skip header
                metadata_json = backup_file.read(metadata.compressed_size_bytes)
                calculated_metadata_hash = hashlib.sha256(metadata_json).hexdigest()
                
                if calculated_metadata_hash != metadata.metadata_checksum:
                    raise VectorStoreError(
                        f"Metadata checksum mismatch in backup: {backup_path}",
                        "integrity_validation_failed"
                    )
                
                # Validate data checksum
                backup_file.seek(16 + len(metadata_json))  # Move to data section
                data_hash = hashlib.sha256()
                
                while True:
                    chunk = backup_file.read(self.backup_chunk_size)
                    if not chunk:
                        break
                    data_hash.update(chunk)
                
                calculated_data_hash = data_hash.hexdigest()
                if calculated_data_hash != metadata.data_checksum:
                    raise VectorStoreError(
                        f"Data checksum mismatch in backup: {backup_path}",
                        "integrity_validation_failed"
                    )
            
            self.logger.info(
                "backup_integrity_validation_passed",
                backup_path=backup_path,
                metadata_checksum=metadata.metadata_checksum[:16],
                data_checksum=metadata.data_checksum[:16]
            )
            return True
            
        except Exception as e:
            self.logger.error(
                "backup_integrity_validation_failed",
                backup_path=backup_path,
                error=str(e)
            )
            return False
    
    async def _extract_backup_data(self, backup_path: str, restore_dir: Path, metadata: BackupMetadata) -> List[str]:
        """Extract backup data to restore directory."""
        try:
            extracted_files = []
            
            with open(backup_path, 'rb') as backup_file:
                # Skip to data section
                backup_file.seek(16 + metadata.compressed_size_bytes)  # Skip header + metadata
                
                # Handle compression
                if metadata.compression_type == CompressionType.GZIP:
                    data_stream = gzip.open(backup_file, 'rb')
                else:
                    data_stream = backup_file
                
                try:
                    # Read file count
                    file_count = struct.unpack('<I', data_stream.read(4))[0]
                    
                    for _ in range(file_count):
                        # Read file metadata
                        name_length = struct.unpack('<I', data_stream.read(4))[0]
                        file_name = data_stream.read(name_length).decode('utf-8')
                        file_size = struct.unpack('<Q', data_stream.read(8))[0]
                        
                        # Extract file content
                        output_file_path = restore_dir / file_name
                        with open(output_file_path, 'wb') as output_file:
                            remaining = file_size
                            while remaining > 0:
                                chunk_size = min(self.backup_chunk_size, remaining)
                                chunk = data_stream.read(chunk_size)
                                if not chunk:
                                    break
                                output_file.write(chunk)
                                remaining -= len(chunk)
                        
                        extracted_files.append(str(output_file_path))
                        
                        self.logger.debug(
                            "file_extracted",
                            file_name=file_name,
                            file_size=file_size,
                            output_path=str(output_file_path)
                        )
                
                finally:
                    if metadata.compression_type == CompressionType.GZIP:
                        data_stream.close()
            
            self.logger.info(
                "backup_data_extraction_completed",
                extracted_files=len(extracted_files),
                restore_dir=str(restore_dir)
            )
            
            return extracted_files
            
        except Exception as e:
            self.logger.error(
                "backup_data_extraction_failed",
                backup_path=backup_path,
                error=str(e)
            )
            raise VectorStoreError(f"Failed to extract backup data: {str(e)}", "extraction_failed")
    
    # Automated Scheduling and Cleanup Methods
    
    async def start_scheduler(self) -> None:
        """Start the automated backup scheduler."""
        if not SCHEDULER_AVAILABLE:
            self.logger.warning("scheduler_not_available", message="apscheduler not installed")
            return
            
        try:
            if not self._scheduler_running and self._scheduler:
                self._scheduler.start()
                self._scheduler_running = True
                
                # Add cleanup job that runs every hour
                self._scheduler.add_job(
                    self._cleanup_old_backups,
                    IntervalTrigger(hours=1),
                    id="cleanup_old_backups",
                    replace_existing=True
                )
                
                self.logger.info("backup_scheduler_started")
            
        except Exception as e:
            self.logger.error("backup_scheduler_start_failed", error=str(e))
            raise VectorStoreError(f"Failed to start backup scheduler: {str(e)}", "scheduler_start_failed")
    
    async def _cleanup_old_backups(self) -> None:
        """Clean up old backups according to retention policy."""
        try:
            self.logger.info("backup_cleanup_started")
            
            all_backups = await self.list_backups()
            if not all_backups:
                return
            
            deleted_count = 0
            
            # Delete backups older than max_age_days
            cutoff_date = datetime.now() - timedelta(days=self._retention_policy.max_age_days)
            
            for backup in all_backups:
                backup_date = datetime.fromisoformat(backup["created_at"])
                
                if backup_date < cutoff_date:
                    await self._delete_backup_file(backup["backup_file_path"])
                    deleted_count += 1
                    
            # Update last cleanup time
            self._last_cleanup = datetime.now().isoformat()
            
            self.logger.info(
                "backup_cleanup_completed",
                deleted_count=deleted_count,
                remaining_backups=len(all_backups) - deleted_count
            )
            
        except Exception as e:
            self.logger.error("backup_cleanup_failed", error=str(e))
    
    async def _delete_backup_file(self, backup_file_path: str) -> None:
        """Delete a backup file."""
        try:
            backup_path = Path(backup_file_path)
            if backup_path.exists():
                backup_path.unlink()
                self.logger.debug("backup_file_deleted", backup_file=backup_file_path)
                
        except Exception as e:
            self.logger.warning(
                "backup_file_deletion_failed",
                backup_file=backup_file_path,
                error=str(e)
            )
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status and active schedules."""
        return {
            "scheduler_available": SCHEDULER_AVAILABLE,
            "running": self._scheduler_running,
            "active_schedules": len(self._schedules),
            "schedules": {
                schedule_id: {
                    "source_path": schedule.source_path,
                    "backup_type": schedule.backup_type.value,
                    "cron_expression": schedule.cron_expression,
                    "enabled": schedule.enabled,
                    "retention_days": schedule.retention_days
                }
                for schedule_id, schedule in self._schedules.items()
            },
            "retention_policy": {
                "max_age_days": self._retention_policy.max_age_days,
                "max_backup_count": self._retention_policy.max_backup_count,
                "min_free_space_gb": self._retention_policy.min_free_space_gb
            },
            "last_cleanup": self._last_cleanup
        }


# Global backup manager instance
backup_manager = VectorBackupManager()