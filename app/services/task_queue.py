"""
Simple task queue service for background processing.
In production, this would be replaced with Redis/Celery or similar.
"""
import asyncio
import json
import uuid
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import traceback

from app.core.logging import get_logger, get_utc_datetime
from app.core.config import settings

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a background task."""
    id: str
    name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


class SimpleTaskQueue:
    """
    Simple in-memory task queue.
    In production, this would be replaced with a proper queue system like Celery.
    """
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.tasks: Dict[str, Task] = {}
        self.pending_tasks: asyncio.Queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self._workers_started = False
        self._shutdown = False
    
    def register_handler(self, task_name: str, handler: Callable):
        """Register a task handler function."""
        self.task_handlers[task_name] = handler
        logger.info("task_handler_registered", task_name=task_name)
    
    async def enqueue(
        self,
        task_name: str,
        *args,
        task_id: Optional[str] = None,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Enqueue a task for background processing.
        
        Args:
            task_name: Name of the task to execute
            args: Positional arguments for the task
            task_id: Optional custom task ID
            max_retries: Maximum number of retry attempts
            kwargs: Keyword arguments for the task
            
        Returns:
            Task ID
        """
        if task_name not in self.task_handlers:
            raise ValueError(f"No handler registered for task: {task_name}")
        
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            name=task_name,
            args=list(args),
            kwargs=kwargs,
            status=TaskStatus.PENDING,
            # REFACTORED: Using existing utility instead of direct datetime.utcnow()
            created_at=get_utc_datetime(),
            max_retries=max_retries
        )
        
        self.tasks[task_id] = task
        await self.pending_tasks.put(task_id)
        
        logger.info(
            "task_enqueued",
            task_id=task_id,
            task_name=task_name,
            queue_size=self.pending_tasks.qsize()
        )
        
        # Start workers if not already started
        if not self._workers_started:
            await self._start_workers()
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get task status by ID."""
        return self.tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            logger.info("task_cancelled", task_id=task_id)
            return True
        elif task.status == TaskStatus.RUNNING and task_id in self.running_tasks:
            # Cancel the asyncio task
            asyncio_task = self.running_tasks[task_id]
            asyncio_task.cancel()
            task.status = TaskStatus.CANCELLED
            logger.info("running_task_cancelled", task_id=task_id)
            return True
        
        return False
    
    async def _start_workers(self):
        """Start background workers."""
        if self._workers_started:
            return
        
        self._workers_started = True
        
        # Start worker tasks
        for i in range(self.max_workers):
            asyncio.create_task(self._worker(f"worker-{i}"))
        
        logger.info("task_queue_workers_started", worker_count=self.max_workers)
    
    async def _worker(self, worker_name: str):
        """Background worker that processes tasks."""
        logger.info("task_worker_started", worker_name=worker_name)
        
        while not self._shutdown:
            try:
                # Get next task (wait up to 1 second)
                task_id = await asyncio.wait_for(
                    self.pending_tasks.get(),
                    timeout=1.0
                )
                
                task = self.tasks.get(task_id)
                if not task or task.status != TaskStatus.PENDING:
                    continue
                
                # Execute task
                await self._execute_task(task, worker_name)
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                logger.error(
                    "worker_error",
                    worker_name=worker_name,
                    error=str(e),
                    exc_info=True
                )
        
        logger.info("task_worker_stopped", worker_name=worker_name)
    
    async def _execute_task(self, task: Task, worker_name: str):
        """Execute a single task."""
        task_id = task.id
        
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            # REFACTORED: Using existing utility instead of direct datetime.utcnow()
            task.started_at = get_utc_datetime()
            
            logger.info(
                "task_execution_started",
                task_id=task_id,
                task_name=task.name,
                worker_name=worker_name,
                retry_count=task.retries
            )
            
            # Get handler
            handler = self.task_handlers[task.name]
            
            # Create asyncio task for execution
            asyncio_task = asyncio.create_task(
                handler(*task.args, **task.kwargs)
            )
            self.running_tasks[task_id] = asyncio_task
            
            try:
                # Execute with timeout (optional)
                result = await asyncio.wait_for(asyncio_task, timeout=300)  # 5 minute timeout
                
                # Task completed successfully
                task.status = TaskStatus.COMPLETED
                # REFACTORED: Using existing utility instead of direct datetime.utcnow()
                task.completed_at = get_utc_datetime()
                task.result = result
                
                logger.info(
                    "task_execution_completed",
                    task_id=task_id,
                    task_name=task.name,
                    worker_name=worker_name,
                    execution_time=(task.completed_at - task.started_at).total_seconds()
                )
                
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                logger.info("task_execution_cancelled", task_id=task_id)
                
            except asyncio.TimeoutError:
                task.status = TaskStatus.FAILED
                task.error = "Task execution timed out"
                logger.error("task_execution_timeout", task_id=task_id)
                
            finally:
                # Remove from running tasks
                self.running_tasks.pop(task_id, None)
                
        except Exception as e:
            # Task execution failed
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
            logger.error(
                "task_execution_failed",
                task_id=task_id,
                task_name=task.name,
                worker_name=worker_name,
                error=str(e),
                traceback=traceback.format_exc()
            )
            
            # Check if we should retry
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                
                # Re-queue with delay
                await asyncio.sleep(min(2 ** task.retries, 60))  # Exponential backoff
                await self.pending_tasks.put(task_id)
                
                logger.info(
                    "task_retry_scheduled",
                    task_id=task_id,
                    retry_count=task.retries,
                    max_retries=task.max_retries
                )
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pending_count = self.pending_tasks.qsize()
        running_count = len(self.running_tasks)
        
        status_counts = {}
        for task in self.tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "pending_tasks": pending_count,
            "running_tasks": running_count,
            "max_workers": self.max_workers,
            "status_counts": status_counts,
            "workers_started": self._workers_started
        }
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up old completed/failed tasks."""
        # REFACTORED: Using existing utility instead of direct datetime.utcnow()
        cutoff_time = get_utc_datetime() - timedelta(hours=max_age_hours)
        
        tasks_to_remove = []
        for task_id, task in self.tasks.items():
            if (
                task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                task.completed_at and
                task.completed_at < cutoff_time
            ):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        
        logger.info("old_tasks_cleaned", removed_count=len(tasks_to_remove))
        return len(tasks_to_remove)
    
    async def shutdown(self):
        """Shutdown the task queue gracefully."""
        self._shutdown = True
        
        # Cancel all running tasks
        for task_id, asyncio_task in self.running_tasks.items():
            asyncio_task.cancel()
            task = self.tasks.get(task_id)
            if task:
                task.status = TaskStatus.CANCELLED
        
        logger.info("task_queue_shutdown")


# Global task queue instance
task_queue = SimpleTaskQueue(max_workers=settings.DEBUG and 1 or 3)


# Task queue management functions
async def enqueue_task(task_name: str, *args, **kwargs) -> str:
    """Enqueue a background task."""
    return await task_queue.enqueue(task_name, *args, **kwargs)


async def get_task_status(task_id: str) -> Optional[Task]:
    """Get task status."""
    return await task_queue.get_task_status(task_id)


async def cancel_task(task_id: str) -> bool:
    """Cancel a task."""
    return await task_queue.cancel_task(task_id)


# Register common task handlers
def register_task_handlers():
    """Register all task handlers."""
    from app.services.document_processor import document_processor
    
    async def process_document_task(document_id: str):
        """Task handler for document processing."""
        await document_processor.process_document_async(document_id)
    
    task_queue.register_handler("process_document", process_document_task)
    
    logger.info("task_handlers_registered")