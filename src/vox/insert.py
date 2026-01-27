"""Platform dispatch for TextInserter."""

from __future__ import annotations

import sys

if sys.platform == "darwin":
    from .insert_darwin import TextInserter
elif sys.platform == "linux":
    from .insert_linux import TextInserter
else:
    raise ImportError(f"Unsupported platform: {sys.platform}")

__all__ = ["TextInserter"]
