#!/usr/bin/env python3
"""
Enhanced CP11 System Launcher

Launches both enhanced API and UI with better coordination and monitoring.
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path
from threading import Thread
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

class EnhancedSystemLauncher:
    """Enhanced system launcher with monitoring and coordination."""
    
    def __init__(self):
        self.api_process = None
        self.ui_process = None
        self.running = True
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n🛑 Shutting down Enhanced NepaliGov-RAG-Bench...")
        self.running = False
        
        if self.api_process:
            self.api_process.terminate()
            print("✅ API server stopped")
        
        if self.ui_process:
            self.ui_process.terminate()
            print("✅ UI server stopped")
        
        sys.exit(0)
    
    def check_api_health(self, max_retries=30, delay=2):
        """Check if API is healthy with retries."""
        for i in range(max_retries):
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get('status') in ['healthy', 'degraded']:
                        return True
            except:
                pass
            
            if not self.running:
                return False
            
            print(f"⏳ Waiting for API to be ready... ({i+1}/{max_retries})")
            time.sleep(delay)
        
        return False
    
    def start_api_server(self):
        """Start the enhanced API server."""
        print("🚀 Starting Enhanced API Server...")
        print("📍 API will be available at: http://localhost:8000")
        print("📚 Enhanced docs at: http://localhost:8000/docs")
        print("📊 Metrics at: http://localhost:8000/metrics")
        
        try:
            self.api_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "api.enhanced_app:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Monitor API startup
            if self.check_api_health():
                print("✅ Enhanced API server is healthy and ready!")
                return True
            else:
                print("❌ API server failed to start properly")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start API server: {e}")
            return False
    
    def start_ui_server(self):
        """Start the enhanced UI server."""
        print("\n🎨 Starting Enhanced Streamlit UI...")
        print("📍 UI will be available at: http://localhost:8501")
        print("🔗 Connected to API at: http://localhost:8000")
        
        try:
            self.ui_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "ui/enhanced_app.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            print("✅ Enhanced UI server started!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start UI server: {e}")
            return False
    
    def monitor_system(self):
        """Monitor system health and performance."""
        print("\n📊 System monitoring started...")
        
        while self.running:
            try:
                # Check API health
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    status = health_data.get('status', 'unknown')
                    
                    if status == 'healthy':
                        status_emoji = "🟢"
                    elif status == 'degraded':
                        status_emoji = "🟡"
                    else:
                        status_emoji = "🔴"
                    
                    # Get metrics
                    metrics_response = requests.get("http://localhost:8000/metrics", timeout=5)
                    if metrics_response.status_code == 200:
                        metrics = metrics_response.json()
                        total_requests = metrics.get('total_requests', 0)
                        avg_time = metrics.get('avg_response_time', 0)
                        cache_hit_rate = metrics.get('cache_hit_rate', 0)
                        
                        print(f"\r{status_emoji} Status: {status} | Requests: {total_requests} | Avg Time: {avg_time:.2f}s | Cache: {cache_hit_rate:.1f}%", end="", flush=True)
                else:
                    print(f"\r🔴 API health check failed (status: {response.status_code})", end="", flush=True)
                    
            except:
                print(f"\r🔴 API unreachable", end="", flush=True)
            
            time.sleep(10)  # Check every 10 seconds
    
    def run(self):
        """Run the enhanced system."""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🚀 Enhanced NepaliGov-RAG-Bench System Launcher")
        print("=" * 60)
        
        # Start API server
        if not self.start_api_server():
            print("❌ Failed to start API server. Exiting.")
            return 1
        
        # Start UI server
        if not self.start_ui_server():
            print("❌ Failed to start UI server. Stopping API and exiting.")
            if self.api_process:
                self.api_process.terminate()
            return 1
        
        print("\n" + "=" * 60)
        print("🎉 Enhanced System is Ready!")
        print("🌐 Open your browser and navigate to:")
        print("   📱 UI: http://localhost:8501")
        print("   🔧 API: http://localhost:8000/docs")
        print("   📊 Metrics: http://localhost:8000/metrics")
        print("\n💡 Features Available:")
        print("   ✨ Advanced caching for faster responses")
        print("   📊 Real-time performance monitoring")
        print("   🎨 Enhanced UI with better visualizations")
        print("   🔍 Improved citation highlighting")
        print("   📈 Analytics dashboard")
        print("   ⚡ Optimized search and answer generation")
        print("\nPress Ctrl+C to stop the system")
        print("=" * 60)
        
        # Start monitoring in a separate thread
        monitor_thread = Thread(target=self.monitor_system, daemon=True)
        monitor_thread.start()
        
        # Wait for processes
        try:
            while self.running:
                if self.api_process and self.api_process.poll() is not None:
                    print("\n❌ API process died unexpectedly")
                    break
                
                if self.ui_process and self.ui_process.poll() is not None:
                    print("\n❌ UI process died unexpectedly")
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
        
        return 0


if __name__ == "__main__":
    launcher = EnhancedSystemLauncher()
    exit_code = launcher.run()
    sys.exit(exit_code)


