#!/usr/bin/env python3
"""Entry point for packaged Vox application."""

import sys
import os

# Add the package to path for frozen app
if getattr(sys, 'frozen', False):
    # Running as compiled
    app_path = os.path.dirname(sys.executable)
    sys.path.insert(0, app_path)

# Import and run the main function
from vox.app import main

if __name__ == "__main__":
    main()
