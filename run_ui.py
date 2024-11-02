#!/usr/bin/env python3
"""
UI Launcher for CP11

Simple launcher script for the Streamlit UI.
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("🎨 Starting NepaliGov-RAG-Bench Streamlit UI...")
    print("📍 UI will be available at: http://localhost:8501")
    print("🔗 Make sure API server is running at: http://localhost:8000")
    print()
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "ui/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])


