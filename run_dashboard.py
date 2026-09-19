#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn
print("""
=== PTAI Local Dashboard ===
Dashboard: http://localhost:8000
No cloud, runs locally on your PC
Shows bankroll, trades, scans, self-preservation status

CLI still works in parallel:
  python main.py run --bankroll 50 --interval 10
""")
uvicorn.run("ptai.dashboard:app", host="0.0.0.0", port=8000, reload=False)
