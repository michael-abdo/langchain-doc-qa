"""
Performance Alerting and Degradation Detection System - Day 3 Reliability Enhancement
Provides real-time performance monitoring with configurable thresholds and alerting mechanisms.

Features:
- Configurable performance degradation thresholds
- Real-time performance tracking with trend analysis
- Multi-level alerting system (warning, critical, emergency)
- Performance trend analysis and prediction
- Automated reporting and notification system
- Threshold management and dynamic adjustment
"""
import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime, timedelta
import statistics
import math
from pathlib import Path

from app.core.common import BaseService, get_service_logger
from app.core.exceptions import ValidationError, VectorStoreError
from app.core.vector_reliability import performance_monitor, PerformanceMetrics


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ThresholdType(Enum):
    """Types of performance thresholds."""
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    TIMEOUT_RATE = "timeout_rate"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    QUEUE_LENGTH = "queue_length"
    MEMORY_USAGE = "memory_usage"


class TrendDirection(Enum):
    """Performance trend directions."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    threshold_id: str
    metric_name: str
    threshold_type: ThresholdType
    warning_value: float
    critical_value: float
    emergency_value: Optional[float] = None
    comparison_operator: str = ">"  # >, <, >=, <=, ==
    evaluation_window_seconds: int = 300  # 5 minutes
    min_samples: int = 10
    enabled: bool = True
    description: str = ""
    remediation_actions: List[str] = field(default_factory=list)


@dataclass
class PerformanceAlert:
    """Performance alert instance."""
    alert_id: str
    timestamp: str
    alert_level: AlertLevel
    threshold_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    trend_analysis: Optional[Dict[str, Any]] = None
    remediation_suggestions: List[str] = field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False
    resolution_timestamp: Optional[str] = None


@dataclass
class TrendAnalysis:
    """Performance trend analysis results."""
    metric_name: str
    window_start: str
    window_end: str
    trend_direction: TrendDirection
    slope: float  # Rate of change
    confidence: float  # Confidence in trend detection (0-1)
    data_points: int
    current_value: float
    predicted_next_value: Optional[float] = None
    volatility: float = 0.0
    significance: float = 0.0  # How significant the trend is


class PerformanceAlerting(BaseService):
    """
    Comprehensive performance alerting and degradation detection system.
    
    Monitors performance metrics in real-time, detects degradation patterns,
    and triggers configurable alerts with trend analysis and remediation guidance.
    """
    
    def __init__(
        self,
        alert_history_size: int = 1000,
        trend_analysis_window: int = 300,  # 5 minutes
        check_interval: float = 30.0  # 30 seconds
    ):
        super().__init__("performance_alerting")
        
        self.alert_history_size = alert_history_size
        self.trend_analysis_window = trend_analysis_window
        self.check_interval = check_interval
        
        # Performance thresholds
        self._thresholds: Dict[str, PerformanceThreshold] = {}
        
        # Alert management
        self._active_alerts: Dict[str, PerformanceAlert] = {}
        self._alert_history: deque = deque(maxlen=alert_history_size)
        self._alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Performance data tracking
        self._metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._last_check_time = 0.0
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Statistics tracking
        self._alert_stats = {
            "total_alerts": 0,
            "active_alerts": 0,
            "resolved_alerts": 0,
            "acknowledged_alerts": 0,
            "alert_counts_by_level": defaultdict(int),
            "alert_counts_by_metric": defaultdict(int)
        }
        
        # Initialize default thresholds
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self) -> None:
        """Set up default performance thresholds."""
        default_thresholds = [
            PerformanceThreshold(
                threshold_id="response_time_database",
                metric_name="database_session",
                threshold_type=ThresholdType.RESPONSE_TIME,
                warning_value=2.0,  # 2 seconds
                critical_value=5.0,  # 5 seconds
                emergency_value=10.0,  # 10 seconds
                description="Database session response time",
                remediation_actions=[
                    "Check database connection pool",
                    "Review slow queries",
                    "Consider connection optimization"
                ]
            ),
            PerformanceThreshold(
                threshold_id="response_time_vector_search",
                metric_name="vector_search",
                threshold_type=ThresholdType.RESPONSE_TIME,
                warning_value=1.0,  # 1 second
                critical_value=3.0,  # 3 seconds
                emergency_value=8.0,  # 8 seconds
                description="Vector search response time",
                remediation_actions=[
                    "Check vector index health",
                    "Review search parameters",
                    "Consider index optimization"
                ]
            ),
            PerformanceThreshold(
                threshold_id="success_rate_general",
                metric_name="*",  # Applies to all metrics
                threshold_type=ThresholdType.SUCCESS_RATE,
                warning_value=95.0,  # 95%
                critical_value=90.0,  # 90%
                emergency_value=80.0,  # 80%
                comparison_operator="<",
                description="General operation success rate",
                remediation_actions=[
                    "Review error logs",
                    "Check service dependencies",
                    "Investigate failure patterns"
                ]
            ),
            PerformanceThreshold(
                threshold_id="timeout_rate_general",
                metric_name="*",
                threshold_type=ThresholdType.TIMEOUT_RATE,
                warning_value=5.0,  # 5%
                critical_value=10.0,  # 10%
                emergency_value=20.0,  # 20%
                description="Operation timeout rate",
                remediation_actions=[
                    "Review timeout configurations",
                    "Check resource availability",
                    "Investigate performance bottlenecks"
                ]
            )
        ]
        
        for threshold in default_thresholds:
            self._thresholds[threshold.threshold_id] = threshold
    
    async def start_monitoring(self) -> None:
        """Start the performance monitoring task."""
        if self._monitoring_task is not None:
            return
        
        self.logger.info("performance_monitoring_started")
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop the performance monitoring task."""
        if self._monitoring_task is None:
            return
        
        self._monitoring_task.cancel()
        try:
            await self._monitoring_task
        except asyncio.CancelledError:
            pass
        
        self._monitoring_task = None
        self.logger.info("performance_monitoring_stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that checks thresholds and generates alerts."""
        while True:
            try:
                await self._check_performance_thresholds()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "performance_monitoring_loop_error",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(self.check_interval)
    
    async def _check_performance_thresholds(self) -> None:
        """Check all configured thresholds and generate alerts if needed."""
        current_time = time.time()
        
        # Get current performance metrics
        all_metrics = performance_monitor.get_all_metrics()
        
        for metric_name, metrics in all_metrics.items():
            # Store metric history
            self._metric_history[metric_name].append({
                "timestamp": current_time,
                "metrics": asdict(metrics)
            })
            
            # Check thresholds for this metric
            await self._evaluate_metric_thresholds(metric_name, metrics, current_time)
        
        # Clean up resolved alerts
        await self._cleanup_resolved_alerts()
        
        self._last_check_time = current_time
    
    async def _evaluate_metric_thresholds(
        self,
        metric_name: str,
        metrics: PerformanceMetrics,
        current_time: float
    ) -> None:
        """Evaluate thresholds for a specific metric."""
        
        # Find applicable thresholds
        applicable_thresholds = []
        for threshold in self._thresholds.values():
            if threshold.enabled and (threshold.metric_name == metric_name or threshold.metric_name == "*"):
                applicable_thresholds.append(threshold)
        
        for threshold in applicable_thresholds:
            await self._check_threshold(threshold, metric_name, metrics, current_time)
    
    async def _check_threshold(
        self,
        threshold: PerformanceThreshold,
        metric_name: str,
        metrics: PerformanceMetrics,
        current_time: float
    ) -> None:
        """Check a specific threshold against current metrics."""
        
        # Get the value to compare based on threshold type
        current_value = self._extract_threshold_value(threshold.threshold_type, metrics)
        if current_value is None:
            return
        
        # Check if we have enough samples
        if metrics.operation_count < threshold.min_samples:
            return
        
        # Determine alert level
        alert_level = None
        threshold_value = None
        
        if self._threshold_exceeded(current_value, threshold.emergency_value, threshold.comparison_operator):
            alert_level = AlertLevel.EMERGENCY
            threshold_value = threshold.emergency_value
        elif self._threshold_exceeded(current_value, threshold.critical_value, threshold.comparison_operator):
            alert_level = AlertLevel.CRITICAL
            threshold_value = threshold.critical_value
        elif self._threshold_exceeded(current_value, threshold.warning_value, threshold.comparison_operator):
            alert_level = AlertLevel.WARNING
            threshold_value = threshold.warning_value
        
        if alert_level is not None:
            # Generate alert
            await self._generate_alert(
                threshold, metric_name, current_value, threshold_value, alert_level, current_time
            )
        else:
            # Check if we should resolve existing alerts
            await self._check_alert_resolution(threshold.threshold_id, current_value, threshold)
    
    def _extract_threshold_value(
        self,
        threshold_type: ThresholdType,
        metrics: PerformanceMetrics
    ) -> Optional[float]:
        """Extract the appropriate value from metrics based on threshold type."""
        
        if threshold_type == ThresholdType.RESPONSE_TIME:
            return metrics.avg_time
        elif threshold_type == ThresholdType.SUCCESS_RATE:
            return metrics.success_rate
        elif threshold_type == ThresholdType.TIMEOUT_RATE:
            if metrics.operation_count > 0:
                return (metrics.timeout_count / metrics.operation_count) * 100.0
            return 0.0
        elif threshold_type == ThresholdType.ERROR_RATE:
            if metrics.operation_count > 0:
                return (metrics.error_count / metrics.operation_count) * 100.0
            return 0.0
        elif threshold_type == ThresholdType.THROUGHPUT:
            # Operations per second (approximate)
            if metrics.total_time > 0:
                return metrics.operation_count / metrics.total_time
            return 0.0
        
        return None
    
    def _threshold_exceeded(
        self,
        current_value: float,
        threshold_value: Optional[float],
        operator: str
    ) -> bool:
        """Check if a threshold is exceeded based on the comparison operator."""
        if threshold_value is None:
            return False
        
        if operator == ">":
            return current_value > threshold_value
        elif operator == "<":
            return current_value < threshold_value
        elif operator == ">=":
            return current_value >= threshold_value
        elif operator == "<=":
            return current_value <= threshold_value
        elif operator == "==":
            return abs(current_value - threshold_value) < 0.001
        
        return False
    
    async def _generate_alert(
        self,
        threshold: PerformanceThreshold,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        alert_level: AlertLevel,
        timestamp: float
    ) -> None:
        """Generate a performance alert."""
        
        alert_key = f"{threshold.threshold_id}_{metric_name}"
        
        # Check if we already have an active alert for this threshold
        if alert_key in self._active_alerts:
            existing_alert = self._active_alerts[alert_key]
            # Update existing alert if severity increased
            if alert_level.value == "emergency" and existing_alert.alert_level != AlertLevel.EMERGENCY:
                existing_alert.alert_level = alert_level
                existing_alert.current_value = current_value
                existing_alert.threshold_value = threshold_value
                existing_alert.timestamp = datetime.fromtimestamp(timestamp).isoformat()
                
                self.logger.error(
                    "performance_alert_escalated",
                    alert_id=existing_alert.alert_id,
                    metric=metric_name,
                    level=alert_level.value
                )
            return
        
        # Generate trend analysis
        trend_analysis = await self._analyze_metric_trend(metric_name, timestamp)
        
        # Create new alert
        alert_id = f"alert_{int(timestamp)}_{threshold.threshold_id}_{metric_name}"
        alert = PerformanceAlert(
            alert_id=alert_id,
            timestamp=datetime.fromtimestamp(timestamp).isoformat(),
            alert_level=alert_level,
            threshold_id=threshold.threshold_id,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            message=self._format_alert_message(threshold, metric_name, current_value, threshold_value, alert_level),
            details={
                "threshold_type": threshold.threshold_type.value,
                "comparison_operator": threshold.comparison_operator,
                "evaluation_window": threshold.evaluation_window_seconds,
                "description": threshold.description
            },
            trend_analysis=asdict(trend_analysis) if trend_analysis else None,
            remediation_suggestions=threshold.remediation_actions.copy()
        )
        
        # Store alert
        self._active_alerts[alert_key] = alert
        self._alert_history.append(alert)
        
        # Update statistics
        self._alert_stats["total_alerts"] += 1
        self._alert_stats["active_alerts"] += 1
        self._alert_stats["alert_counts_by_level"][alert_level.value] += 1
        self._alert_stats["alert_counts_by_metric"][metric_name] += 1
        
        # Log alert
        log_level = "error" if alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY] else "warning"
        getattr(self.logger, log_level)(
            "performance_alert_generated",
            alert_id=alert.alert_id,
            metric=metric_name,
            level=alert_level.value,
            current_value=current_value,
            threshold_value=threshold_value,
            trend_direction=trend_analysis.trend_direction.value if trend_analysis else "unknown"
        )
        
        # Trigger alert callbacks
        await self._trigger_alert_callbacks(alert)
    
    def _format_alert_message(
        self,
        threshold: PerformanceThreshold,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        alert_level: AlertLevel
    ) -> str:
        """Format a human-readable alert message."""
        
        threshold_type_names = {
            ThresholdType.RESPONSE_TIME: "response time",
            ThresholdType.SUCCESS_RATE: "success rate",
            ThresholdType.TIMEOUT_RATE: "timeout rate",
            ThresholdType.ERROR_RATE: "error rate",
            ThresholdType.THROUGHPUT: "throughput"
        }
        
        threshold_name = threshold_type_names.get(threshold.threshold_type, threshold.threshold_type.value)
        
        units = {
            ThresholdType.RESPONSE_TIME: "seconds",
            ThresholdType.SUCCESS_RATE: "%",
            ThresholdType.TIMEOUT_RATE: "%",
            ThresholdType.ERROR_RATE: "%",
            ThresholdType.THROUGHPUT: "ops/sec"
        }
        
        unit = units.get(threshold.threshold_type, "")
        
        return (
            f"[{alert_level.value.upper()}] {metric_name} {threshold_name} "
            f"{threshold.comparison_operator} {threshold_value}{unit}: "
            f"current value {current_value:.2f}{unit}"
        )
    
    async def _analyze_metric_trend(
        self,
        metric_name: str,
        current_time: float
    ) -> Optional[TrendAnalysis]:
        """Analyze performance trends for a metric."""
        
        history = self._metric_history.get(metric_name)
        if not history or len(history) < 5:
            return None
        
        # Get data points within trend analysis window
        window_start = current_time - self.trend_analysis_window
        relevant_points = [
            point for point in history
            if point["timestamp"] >= window_start
        ]
        
        if len(relevant_points) < 5:
            return None
        
        try:
            # Extract values for trend analysis (using average response time)
            timestamps = [point["timestamp"] for point in relevant_points]
            values = [point["metrics"]["avg_time"] for point in relevant_points]
            
            # Calculate trend slope using linear regression
            n = len(values)
            sum_x = sum(range(n))
            sum_y = sum(values)
            sum_xy = sum(i * y for i, y in enumerate(values))
            sum_x2 = sum(i * i for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Calculate correlation coefficient for confidence
            mean_x = sum_x / n
            mean_y = sum_y / n
            
            numerator = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(values))
            denominator_x = sum((i - mean_x) ** 2 for i in range(n))
            denominator_y = sum((y - mean_y) ** 2 for y in values)
            
            if denominator_x > 0 and denominator_y > 0:
                correlation = numerator / math.sqrt(denominator_x * denominator_y)
                confidence = abs(correlation)
            else:
                confidence = 0.0
            
            # Determine trend direction
            if abs(slope) < 0.001 and confidence < 0.3:
                trend_direction = TrendDirection.STABLE
            elif confidence < 0.2:
                trend_direction = TrendDirection.VOLATILE
            elif slope > 0.01:
                trend_direction = TrendDirection.DEGRADING
            elif slope < -0.01:
                trend_direction = TrendDirection.IMPROVING
            else:
                trend_direction = TrendDirection.STABLE
            
            # Calculate volatility
            if len(values) > 1:
                volatility = statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) > 0 else 0.0
            else:
                volatility = 0.0
            
            # Predict next value
            predicted_next = values[-1] + slope if confidence > 0.5 else None
            
            return TrendAnalysis(
                metric_name=metric_name,
                window_start=datetime.fromtimestamp(window_start).isoformat(),
                window_end=datetime.fromtimestamp(current_time).isoformat(),
                trend_direction=trend_direction,
                slope=slope,
                confidence=confidence,
                data_points=len(relevant_points),
                current_value=values[-1],
                predicted_next_value=predicted_next,
                volatility=volatility,
                significance=abs(slope) * confidence
            )
            
        except Exception as e:
            self.logger.warning(
                "trend_analysis_failed",
                metric=metric_name,
                error=str(e)
            )
            return None
    
    async def _check_alert_resolution(
        self,
        threshold_id: str,
        current_value: float,
        threshold: PerformanceThreshold
    ) -> None:
        """Check if any alerts should be resolved based on current values."""
        
        alerts_to_resolve = []
        for alert_key, alert in self._active_alerts.items():
            if alert.threshold_id == threshold_id:
                # Check if current value is now within acceptable range
                if not self._threshold_exceeded(current_value, threshold.warning_value, threshold.comparison_operator):
                    alerts_to_resolve.append(alert_key)
        
        # Resolve alerts
        for alert_key in alerts_to_resolve:
            alert = self._active_alerts.pop(alert_key)
            alert.resolved = True
            alert.resolution_timestamp = datetime.now().isoformat()
            
            self._alert_stats["active_alerts"] -= 1
            self._alert_stats["resolved_alerts"] += 1
            
            self.logger.info(
                "performance_alert_resolved",
                alert_id=alert.alert_id,
                metric=alert.metric_name,
                resolution_value=current_value
            )
    
    async def _cleanup_resolved_alerts(self) -> None:
        """Clean up old resolved alerts from active alerts."""
        # This is handled in _check_alert_resolution, but we could add
        # additional cleanup logic here if needed (e.g., time-based cleanup)
        pass
    
    async def _trigger_alert_callbacks(self, alert: PerformanceAlert) -> None:
        """Trigger all registered alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(
                    "alert_callback_failed",
                    callback=str(callback),
                    alert_id=alert.alert_id,
                    error=str(e)
                )
    
    # Public API methods
    
    def add_threshold(self, threshold: PerformanceThreshold) -> None:
        """Add a new performance threshold."""
        self._thresholds[threshold.threshold_id] = threshold
        self.logger.info(
            "performance_threshold_added",
            threshold_id=threshold.threshold_id,
            metric=threshold.metric_name,
            type=threshold.threshold_type.value
        )
    
    def remove_threshold(self, threshold_id: str) -> bool:
        """Remove a performance threshold."""
        if threshold_id in self._thresholds:
            del self._thresholds[threshold_id]
            self.logger.info("performance_threshold_removed", threshold_id=threshold_id)
            return True
        return False
    
    def get_threshold(self, threshold_id: str) -> Optional[PerformanceThreshold]:
        """Get a specific threshold."""
        return self._thresholds.get(threshold_id)
    
    def list_thresholds(self) -> List[PerformanceThreshold]:
        """List all configured thresholds."""
        return list(self._thresholds.values())
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add an alert callback function."""
        self._alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active alerts."""
        return list(self._active_alerts.values())
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[PerformanceAlert]:
        """Get alert history."""
        if limit is None:
            return list(self._alert_history)
        return list(self._alert_history)[-limit:]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._active_alerts.values():
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self._alert_stats["acknowledged_alerts"] += 1
                self.logger.info("performance_alert_acknowledged", alert_id=alert_id)
                return True
        return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        return {
            **self._alert_stats,
            "threshold_count": len(self._thresholds),
            "monitoring_active": self._monitoring_task is not None,
            "last_check_time": datetime.fromtimestamp(self._last_check_time).isoformat() if self._last_check_time > 0 else None
        }
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        current_time = time.time()
        
        # Analyze trends for all metrics
        trend_analyses = {}
        for metric_name in self._metric_history.keys():
            trend = await self._analyze_metric_trend(metric_name, current_time)
            if trend:
                trend_analyses[metric_name] = asdict(trend)
        
        # Get current performance metrics
        current_metrics = performance_monitor.get_all_metrics()
        
        return {
            "report_timestamp": datetime.fromtimestamp(current_time).isoformat(),
            "alert_statistics": self.get_alert_statistics(),
            "active_alerts": [asdict(alert) for alert in self.get_active_alerts()],
            "trend_analyses": trend_analyses,
            "current_metrics": {name: asdict(metrics) for name, metrics in current_metrics.items()},
            "threshold_summary": {
                "total_thresholds": len(self._thresholds),
                "enabled_thresholds": sum(1 for t in self._thresholds.values() if t.enabled),
                "threshold_types": list(set(t.threshold_type.value for t in self._thresholds.values()))
            }
        }


# Global instance for shared use
performance_alerting = PerformanceAlerting()