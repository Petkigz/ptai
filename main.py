#!/usr/bin/env python3
"""
PTAI Main Entry - for direct python main.py execution
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ptai.cli import app

if __name__ == "__main__":
    app()
