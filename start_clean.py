#!/usr/bin/env python3
"""
Clean Startup Script - Ensures clean SAS environment before starting

This script should be run before starting any SAS-dependent applications
to prevent orphaned session issues.
"""

import subprocess
import sys
import os
import time
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

def main():
    """Main cleanup function"""
    print("🚀 SAS Environment Cleanup")
    print("=" * 40)
    
    # Clean up SAS sessions
    cleanup_sas_sessions()
    
    # Check port availability
    check_port_availability()
    
    print("\n✅ Environment ready for SAS applications")
    print("\nYou can now start your Streamlit application:")
    print("streamlit run your_app.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 