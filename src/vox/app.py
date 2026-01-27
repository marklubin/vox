"""Platform dispatch for Vox application."""

from __future__ import annotations

import sys

from .config import get_logger

log = get_logger("app")


def main() -> None:
    """Entry point for Vox application."""
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
