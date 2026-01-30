"""
TeeLogger - Duplicate stdout/stderr to console and file

This module provides a TeeLogger class that captures all console output
and writes it to both the terminal and a timestamped log file.

Usage:
    from src.utils.tee_logger import TeeLogger

    with TeeLogger("training", "train_chimney"):
        print("This goes to console AND log file")
        # ... rest of script
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


class TeeStream:
    """Stream that writes to multiple outputs simultaneously."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def fileno(self) -> int:
        """Return file descriptor of primary stream (for tqdm compatibility)."""
        return self.streams[0].fileno()

    def isatty(self) -> bool:
        """Check if primary stream is a TTY (for tqdm compatibility)."""
        return self.streams[0].isatty()


class TeeLogger:
    """
    Context manager that duplicates stdout/stderr to a log file.

    Creates timestamped log files organized by category.
    Preserves all console output while saving a permanent record.

    Args:
        category: Log category (e.g., "training", "inference")
        script_name: Name of the script (used in log filename)
        logs_dir: Base directory for logs (default: PROJECT_ROOT/logs)
        capture_stderr: Whether to also capture stderr (default: True)

    Example:
        with TeeLogger("training", "train_chimney_detector"):
            print("Training started...")
            # All print output goes to both console and log file
    """

    def __init__(
        self,
        category: str,
        script_name: str,
        logs_dir: Optional[Path] = None,
        experiment_dir: Optional[Path] = None,
        capture_stderr: bool = True
    ):
        self.category = category
        self.script_name = script_name
        self.capture_stderr = capture_stderr

        # If experiment_dir provided, write logs to experiment_dir/logs/
        if experiment_dir is not None:
            self.logs_dir = Path(experiment_dir) / "logs"
        elif logs_dir is None:
            from ..config import LOGS_DIR
            self.logs_dir = Path(LOGS_DIR) / category
        else:
            self.logs_dir = Path(logs_dir) / category

        self._original_stdout = None
        self._original_stderr = None
        self._log_file = None
        self._log_path = None

    def _generate_log_filename(self) -> str:
        """Generate timestamped log filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.script_name}_{timestamp}.txt"

    def __enter__(self) -> "TeeLogger":
        """Start capturing output to log file."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._log_path = self.logs_dir / self._generate_log_filename()
        self._log_file = open(self._log_path, 'w', encoding='utf-8')

        # Write header
        self._log_file.write(f"{'='*60}\n")
        self._log_file.write(f"Log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._log_file.write(f"Script: {self.script_name}\n")
        self._log_file.write(f"Category: {self.category}\n")
        self._log_file.write(f"{'='*60}\n\n")
        self._log_file.flush()

        # Store and replace stdout
        self._original_stdout = sys.stdout
        sys.stdout = TeeStream(self._original_stdout, self._log_file)

        # Optionally capture stderr
        if self.capture_stderr:
            self._original_stderr = sys.stderr
            sys.stderr = TeeStream(self._original_stderr, self._log_file)

        print(f"[TeeLogger] Logging to: {self._log_path}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Stop capturing and close log file."""
        # Write footer
        if self._log_file:
            self._log_file.write(f"\n{'='*60}\n")
            self._log_file.write(f"Log ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if exc_type:
                self._log_file.write(f"Exception: {exc_type.__name__}: {exc_val}\n")
            self._log_file.write(f"{'='*60}\n")

        # Restore original streams
        if self._original_stdout:
            sys.stdout = self._original_stdout
        if self._original_stderr:
            sys.stderr = self._original_stderr

        # Close log file
        if self._log_file:
            self._log_file.close()

        # Print log location to console
        if self._log_path and self._original_stdout:
            self._original_stdout.write(f"\n[TeeLogger] Log saved to: {self._log_path}\n")

        return False  # Don't suppress exceptions

    @property
    def log_path(self) -> Optional[Path]:
        """Return the path to the current log file."""
        return self._log_path
