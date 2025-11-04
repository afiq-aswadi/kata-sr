"""Pytest configuration for test discovery."""

import sys
from pathlib import Path

# add parent directory to path so tests can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
