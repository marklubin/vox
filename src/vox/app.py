"""Platform dispatch for Vox application."""

from __future__ import annotations

import fcntl
import os
import sys
from pathlib import Path

from .config import get_logger

log = get_logger("app")

# Singleton lock file
_lock_file = None
_lock_fd = None


def _acquire_singleton_lock() -> bool:
    """Acquire singleton lock to prevent multiple instances.

    Returns:
        True if lock acquired, False if another instance is running.
    """
    global _lock_file, _lock_fd

    # Use XDG runtime dir or fall back to /tmp
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
    _lock_file = Path(runtime_dir) / "vox.lock"

    try:
        _lock_fd = open(_lock_file, "w")
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd.write(str(os.getpid()))
        _lock_fd.flush()
        return True
    except (IOError, OSError):
        log.error("Another instance of Vox is already running")
        return False


def main() -> None:
    """Entry point for Vox application."""
    if not _acquire_singleton_lock():
        print("Vox is already running. Exiting.")
        sys.exit(1)

    log.info("=" * 60)
    log.info("Vox starting up")
    log.info("Platform: %s", sys.platform)
    log.info("=" * 60)

    if sys.platform == "darwin":
        from .app_darwin import VoxApp
    elif sys.platform == "linux":
        from .app_linux import VoxApp
    else:
        print(f"Unsupported platform: {sys.platform}")
        sys.exit(1)

    app = VoxApp()
    app.run()


if __name__ == "__main__":
    main()
