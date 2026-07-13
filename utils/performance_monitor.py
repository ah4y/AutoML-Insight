"""Performance monitoring utilities for AutoML-Insight."""

import functools
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List

import psutil

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Track performance metrics for the application."""

    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []
        self.process = psutil.Process(os.getpid())

    def record_metric(
        self, function_name: str, duration: float, memory_before: float, memory_after: float, **kwargs
    ) -> None:
        """Record a performance metric."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "function": function_name,
            "duration_seconds": round(duration, 3),
            "memory_before_mb": round(memory_before, 2),
            "memory_after_mb": round(memory_after, 2),
            "memory_delta_mb": round(memory_after - memory_before, 2),
            **kwargs,
        }
        self.metrics.append(metric)
        logger.info(
            f"Performance: {function_name} took {duration:.2f}s, "
            f"memory delta: {memory_after - memory_before:+.2f}MB"
        )

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all recorded metrics."""
        return self.metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics:
            return {}

        durations = [m["duration_seconds"] for m in self.metrics]
        memory_deltas = [m["memory_delta_mb"] for m in self.metrics]

        return {
            "total_calls": len(self.metrics),
            "total_duration": sum(durations),
            "avg_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "total_memory_delta": sum(memory_deltas),
            "avg_memory_delta": sum(memory_deltas) / len(memory_deltas),
            "max_memory_delta": max(memory_deltas),
        }

    def export_metrics(self, filepath: str) -> None:
        """Export metrics to JSON file."""
        try:
            with open(filepath, "w") as f:
                json.dump({"metrics": self.metrics, "summary": self.get_summary()}, f, indent=2)
            logger.info(f"Exported {len(self.metrics)} metrics to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    def clear_metrics(self) -> None:
        """Clear all recorded metrics."""
        self.metrics.clear()
        logger.info("Cleared all performance metrics")

    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024


# Global performance monitor instance
_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _monitor


def monitor_performance(track_memory: bool = True):
    """
    Decorator to monitor function performance.

    Args:
        track_memory: Whether to track memory usage

    Usage:
        @monitor_performance()
        def my_function(arg1, arg2):
            # function code
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            monitor = get_performance_monitor()

            # Record start state
            start_time = time.time()
            memory_before = monitor.get_current_memory_mb() if track_memory else 0

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Record end state
                end_time = time.time()
                memory_after = monitor.get_current_memory_mb() if track_memory else 0
                duration = end_time - start_time

                # Record metric
                monitor.record_metric(
                    function_name=func.__name__,
                    duration=duration,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    success=True,
                )

                return result

            except Exception as e:
                # Record failure
                end_time = time.time()
                memory_after = monitor.get_current_memory_mb() if track_memory else 0
                duration = end_time - start_time

                monitor.record_metric(
                    function_name=func.__name__,
                    duration=duration,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    success=False,
                    error=str(e),
                )

                raise

        return wrapper

    return decorator


def monitor_section(section_name: str):
    """
    Context manager for monitoring code sections.

    Usage:
        with monitor_section("data_preprocessing"):
            # code to monitor
            pass
    """

    class SectionMonitor:
        def __init__(self, name: str):
            self.name = name
            self.monitor = get_performance_monitor()
            self.start_time = None
            self.memory_before = None

        def __enter__(self):
            self.start_time = time.time()
            self.memory_before = self.monitor.get_current_memory_mb()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            end_time = time.time()
            memory_after = self.monitor.get_current_memory_mb()
            duration = end_time - self.start_time

            self.monitor.record_metric(
                function_name=self.name,
                duration=duration,
                memory_before=self.memory_before,
                memory_after=memory_after,
                success=exc_type is None,
                error=str(exc_val) if exc_val else None,
            )

    return SectionMonitor(section_name)


class PerformanceTimer:
    """Simple timer for measuring execution time."""

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.elapsed = 0

    def start(self) -> "PerformanceTimer":
        """Start the timer."""
        self.start_time = time.time()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            logger.warning(f"Timer '{self.name}' was not started")
            return 0

        self.elapsed = time.time() - self.start_time
        logger.info(f"{self.name}: {self.elapsed:.3f}s")
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def get_system_info() -> Dict[str, Any]:
    """Get current system performance information."""
    process = psutil.Process(os.getpid())

    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "process_memory_mb": process.memory_info().rss / (1024**2),
        "process_cpu_percent": process.cpu_percent(),
        "num_threads": process.num_threads(),
    }


def log_system_info() -> None:
    """Log current system information."""
    info = get_system_info()
    logger.info(
        f"System Info: CPU={info['cpu_percent']}%, "
        f"Memory={info['memory_percent']}% "
        f"({info['memory_available_gb']:.2f}GB available), "
        f"Process={info['process_memory_mb']:.2f}MB"
    )
