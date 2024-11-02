#!/usr/bin/env python3
"""
🚀 Reliable startup script for Nepal Government Q&A with Nepali Keyboard
"""

import os
import sys
import socket
import time
import subprocess
from pathlib import Path

def find_available_port(start_port=8090, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    return None

def kill_existing_processes():
    """Kill any existing web app processes."""
    try:
        # Kill by process name
        subprocess.run(['pkill', '-f', 'enhanced_web_app.py'], capture_output=True)
        subprocess.run(['pkill', '-f', 'simple_web_app.py'], capture_output=True)
        time.sleep(2)
        
        # Kill by port
        for port in range(8090, 8100):
            try:
                result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                      capture_output=True, text=True)
                if result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        subprocess.run(['kill', '-9', pid], capture_output=True)
            except:
                pass
        
        print("✅ Cleaned up existing processes")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

def main():
    """Main startup function."""
    print("🇳🇵 Nepal Government Q&A - Enhanced Startup")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Kill existing processes
    kill_existing_processes()
    
    # Find available port
    port = find_available_port()
    if not port:
        print("❌ Could not find available port!")
        sys.exit(1)
    
    print(f"🔍 Found available port: {port}")
    
    # Check if corpus exists
    corpus_file = project_dir / "data" / "real_corpus.parquet"
    if not corpus_file.exists():
        print("❌ Corpus file not found! Please ensure data/real_corpus.parquet exists.")
        sys.exit(1)
    
    # Start the enhanced web app
    print(f"🚀 Starting Enhanced Nepal Government Q&A...")
    print(f"📍 URL: http://localhost:{port}")
    print(f"🎯 Features: Nepali Keyboard, Language Selection, Advanced UI")
    print()
    
    try:
        # Import and start the app
        sys.path.insert(0, str(project_dir))
        
        # Modify the enhanced_web_app to use our port
        with open('enhanced_web_app.py', 'r') as f:
            content = f.read()
        
        # Replace the start_server call to use our port
        modified_content = content.replace(
            'start_server(8090)',
            f'start_server({port})'
        )
        
        # Execute the modified content
        exec(modified_content)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
