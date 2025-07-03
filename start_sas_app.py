#!/usr/bin/env python3
"""
SAS Application Startup Script

This script ensures a clean SAS environment before starting any SAS-dependent application.
It should be run before starting Streamlit apps or other SAS applications.

Usage:
    python start_sas_app.py [app_name]
    
Examples:
    python start_sas_app.py sgsnm_v3.py
    python start_sas_app.py simple_GLM_SAS_ncss_manager.py
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

def cleanup_sas_sessions():
    """Clean up any existing SAS sessions"""
    print("🧹 Cleaning up existing SAS sessions...")
    
    try:
        # Kill any existing saspy2j processes
        result = subprocess.run(
            ['pkill', '-f', 'saspy2j'], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ Cleaned up existing SAS sessions")
        else:
            print("ℹ️  No existing SAS sessions found")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up SAS sessions: {e}")

def check_port_availability(port=8501):
    """Check if Streamlit port is available"""
    try:
        result = subprocess.run(
            ['lsof', '-i', f':{port}'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        
        if result.stdout.strip():
            print(f"⚠️  Port {port} is in use. Attempting to free it...")
            # Kill processes using the port
            subprocess.run(['lsof', '-ti', f':{port}', '|', 'xargs', 'kill', '-9'], 
                         shell=True, capture_output=True)
            time.sleep(2)
            return True
        else:
            print(f"✅ Port {port} is available")
            return True
            
    except Exception as e:
        print(f"⚠️  Could not check port {port}: {e}")
        return True

def run_session_monitor():
    """Run the session monitor to check current status"""
    try:
        print("🔍 Checking current SAS session status...")
        result = subprocess.run(
            ['python', 'utils/sas_session_monitor.py'], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ Session monitor check completed")
            # Extract key info from output
            for line in result.stdout.split('\n'):
                if 'Current SAS sessions:' in line:
                    print(f"   {line.strip()}")
                elif 'Total CPU usage:' in line:
                    print(f"   {line.strip()}")
                elif 'Total memory usage:' in line:
                    print(f"   {line.strip()}")
        else:
            print("⚠️  Session monitor check failed")
            
    except Exception as e:
        print(f"⚠️  Could not run session monitor: {e}")

def start_streamlit_app(app_name):
    """Start the specified Streamlit application"""
    if not os.path.exists(app_name):
        print(f"❌ Application file not found: {app_name}")
        return False
    
    print(f"🚀 Starting Streamlit application: {app_name}")
    try:
        # Start Streamlit in the background
        process = subprocess.Popen(
            ['streamlit', 'run', app_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"✅ Streamlit started with PID: {process.pid}")
        print(f"🌐 Application should be available at: http://localhost:8501")
        print(f"📝 To stop the application, press Ctrl+C or kill process {process.pid}")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return False

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description='Start SAS application with clean environment')
    parser.add_argument('app_name', nargs='?', help='Streamlit app to start (e.g., sgsnm_v3.py)')
    parser.add_argument('--port', type=int, default=8501, help='Port to use (default: 8501)')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip cleanup step')
    parser.add_argument('--monitor-only', action='store_true', help='Only run session monitor')
    
    args = parser.parse_args()
    
    print("🚀 SAS Application Startup")
    print("=" * 40)
    
    # Run session monitor if requested
    if args.monitor_only:
        run_session_monitor()
        return 0
    
    # Clean up SAS sessions
    if not args.no_cleanup:
        cleanup_sas_sessions()
    
    # Check port availability
    check_port_availability(args.port)
    
    # Run session monitor
    run_session_monitor()
    
    print("\n✅ Environment ready for SAS applications")
    
    # Start application if specified
    if args.app_name:
        process = start_streamlit_app(args.app_name)
        if process:
            try:
                # Wait for the process to complete
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping application...")
                process.terminate()
                process.wait()
                print("✅ Application stopped")
        return 0
    else:
        print("\nYou can now start your Streamlit application:")
        print("streamlit run your_app.py")
        print("\nOr use this script with an app name:")
        print("python start_sas_app.py sgsnm_v3.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 