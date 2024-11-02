#!/usr/bin/env python3
"""
API Server Launcher for CP11

Simple launcher script for the FastAPI backend.
"""

import uvicorn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("🚀 Starting NepaliGov-RAG-Bench API Server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 Interactive docs at: http://localhost:8000/docs")
    print("🔍 Health check at: http://localhost:8000/health")
    print()
    
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


