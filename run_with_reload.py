#!/usr/bin/env python3
"""
Auto-reload launcher for enhanced_web_app.py
Watches for file changes and automatically restarts the server.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / 'venv' / 'bin' / 'python'
SERVER_SCRIPT = PROJECT_ROOT / 'enhanced_web_app.py'


def kill_existing_servers():
    """Kill any existing server processes."""
    try:
        subprocess.run(['pkill', '-f', 'enhanced_web_app.py'], stderr=subprocess.DEVNULL)
        time.sleep(1)
    except Exception:
        pass


def start_server():
    """Start the server process and watch for changes."""
    kill_existing_servers()

    print("🚀 Starting Enhanced Nepal Gov Q&A Server with Auto-Reload...")
    print("👁️  Watching for file changes...")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 50)

    server_process = None

    try:
        last_modified = get_file_modified_time()

        while True:
            # Ensure working directory
            os.chdir(PROJECT_ROOT)

            # Choose python interpreter
            python_exec = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

            # Start the server
            server_process = subprocess.Popen(
                [python_exec, str(SERVER_SCRIPT)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            while True:
                time.sleep(1)

                # Stream output if available
                if server_process.stdout and not server_process.stdout.closed:
                    try:
                        while True:
                            line = server_process.stdout.readline()
                            if not line:
                                break
                            print(line.rstrip())
                    except Exception:
                        pass

                # If process exited, break
                if server_process.poll() is not None:
                    break

                # Detect file changes
                current_modified = get_file_modified_time()
                if current_modified > last_modified:
                    print("\n🔄 File changes detected - restarting server...")
                    last_modified = current_modified
                    server_process.terminate()
                    try:
                        server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server_process.kill()
                    time.sleep(1)
                    break  # Restart loop

            # If the process ended without file change, exit
            if get_file_modified_time() == last_modified and server_process.poll() is not None:
                print("❌ Server process ended. Not restarting (no file changes).")
                break

    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
    finally:
        if server_process and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
        kill_existing_servers()
        print("👋 Server stopped")


def get_file_modified_time():
    """Get the latest modification time of watched files."""
    watch_files = [
        PROJECT_ROOT / 'enhanced_web_app.py',
        PROJECT_ROOT / 'data' / 'pdf_titles.json',
        PROJECT_ROOT / 'api' / 'translation.py'
    ]

    latest = 0.0
    for file_path in watch_files:
        try:
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                latest = max(latest, mtime)
        except Exception:
            continue

    return latest


if __name__ == "__main__":
    start_server()
