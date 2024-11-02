#!/usr/bin/env python3
"""
🇳🇵 Nepal Government Q&A System Launcher

Starts both the API and UI components.
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_corpus():
    """Check if the real corpus exists."""
    corpus_path = Path("data/real_corpus.parquet")
    if not corpus_path.exists():
        print("❌ Real corpus not found!")
        print("🔧 Please run: python create_real_corpus.py")
        return False
    
    print(f"✅ Real corpus found: {corpus_path}")
    return True

def test_api(port=8000, timeout=30):
    """Test if API is responding."""
    for i in range(timeout):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ API is healthy on port {port}")
                return True
        except:
            pass
        
        if i == 0:
            print(f"⏳ Waiting for API on port {port}...")
        time.sleep(1)
    
    print(f"❌ API not responding on port {port}")
    return False

def start_api():
    """Start the API server."""
    print("🚀 Starting Nepal Government Q&A API...")
    
    try:
        # Start API in background
        process = subprocess.Popen([
            sys.executable, "nepal_gov_api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for API to start
        if test_api():
            print("✅ API started successfully!")
            return process
        else:
            print("❌ API failed to start")
            process.terminate()
            return None
            
    except Exception as e:
        print(f"❌ Error starting API: {e}")
        return None

def start_ui():
    """Start the Streamlit UI."""
    print("🚀 Starting Web Interface...")
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "nepal_gov_ui.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error starting UI: {e}")

def main():
    """Main launcher."""
    print("🇳🇵 Nepal Government Q&A System")
    print("=" * 50)
    
    # Check prerequisites
    if not check_corpus():
        return
    
    # Start API
    api_process = start_api()
    if not api_process:
        return
    
    try:
        # Start UI (this will block)
        start_ui()
    finally:
        # Clean up
        if api_process:
            print("🛑 Stopping API server...")
            api_process.terminate()
            api_process.wait()
        print("✅ System stopped")

if __name__ == "__main__":
    main()


