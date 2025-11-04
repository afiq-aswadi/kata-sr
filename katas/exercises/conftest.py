"""Pytest configuration for kata exercises."""

import sys
from pathlib import Path

# add parent directory (katas/) to path so exercises can import framework
sys.path.insert(0, str(Path(__file__).parent.parent))
