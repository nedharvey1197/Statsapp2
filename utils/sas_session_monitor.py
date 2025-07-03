"""
SAS Session Monitor - Prevents orphaned SAS sessions

This utility monitors and cleans up orphaned SAS Java processes
that can accumulate and cause system performance issues.
"""

import subprocess
import logging
import time
import signal
import os
from typing import List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class SASSessionMonitor:
    """Monitors and manages SAS session processes"""
    
    def __init__(self):
        self.sas_processes = {}
        self.max_session_age = 3600  # 1 hour in seconds
    
    def find_sas_processes(self) -> List[Dict]:
        """Find all running SAS Java processes"""
        try:
            result = subprocess.run(
                ['ps', 'aux'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            sas_processes = []
            for line in result.stdout.split('\n'):
                if 'saspy2j' in line and 'java' in line:
                    parts = line.split()
                    if len(parts) >= 10:
                        process_info = {
                            'pid': int(parts[1]),
                            'cpu': float(parts[2]),
                            'mem': float(parts[3]),
                            'time': parts[8],
                            'command': ' '.join(parts[10:]),
                            'start_time': self._parse_start_time(parts[8])
                        }
                        sas_processes.append(process_info)
            
            return sas_processes
            
        except Exception as e:
            logger.error(f"Error finding SAS processes: {e}")
            return []
    
    def _parse_start_time(self, time_str: str) -> datetime:
        """Parse process start time"""
        try:
            # Handle different time formats
            if ':' in time_str:
                # Format: HH:MM:SS
                hours, minutes, seconds = map(int, time_str.split(':'))
                total_seconds = hours * 3600 + minutes * 60 + seconds
                return datetime.now().replace(
                    hour=datetime.now().hour - (total_seconds // 3600),
                    minute=datetime.now().minute - ((total_seconds % 3600) // 60)
                )
            else:
                # Format: MM:SS
                minutes, seconds = map(int, time_str.split(':'))
                total_seconds = minutes * 60 + seconds
                return datetime.now().replace(
                    minute=datetime.now().minute - (total_seconds // 60)
                )
        except:
            return datetime.now()
    
    def cleanup_old_sessions(self, max_age_seconds: int = None) -> int:
        """Clean up SAS sessions older than specified age"""
        if max_age_seconds is None:
            max_age_seconds = self.max_session_age
        
        sas_processes = self.find_sas_processes()
        cleaned_count = 0
        
        for process in sas_processes:
            age_seconds = (datetime.now() - process['start_time']).total_seconds()
            
            if age_seconds > max_age_seconds:
                logger.warning(f"Cleaning up old SAS process {process['pid']} (age: {age_seconds:.0f}s)")
                try:
                    os.kill(process['pid'], signal.SIGTERM)
                    time.sleep(1)
                    
                    # Check if process is still running
                    try:
                        os.kill(process['pid'], 0)
                        # Process still running, force kill
                        os.kill(process['pid'], signal.SIGKILL)
                        logger.info(f"Force killed SAS process {process['pid']}")
                    except OSError:
                        # Process already terminated
                        pass
                    
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to kill SAS process {process['pid']}: {e}")
        
        return cleaned_count
    
    def get_session_stats(self) -> Dict:
        """Get statistics about current SAS sessions"""
        sas_processes = self.find_sas_processes()
        
        total_cpu = sum(p['cpu'] for p in sas_processes)
        total_mem = sum(p['mem'] for p in sas_processes)
        
        return {
            'total_sessions': len(sas_processes),
            'total_cpu_percent': total_cpu,
            'total_mem_percent': total_mem,
            'oldest_session_age': max(
                [(datetime.now() - p['start_time']).total_seconds() for p in sas_processes] + [0]
            ),
            'processes': sas_processes
        }
    
    def emergency_cleanup(self) -> int:
        """Emergency cleanup of all SAS processes"""
        logger.warning("Performing emergency cleanup of all SAS processes")
        
        try:
            result = subprocess.run(
                ['pkill', '-f', 'saspy2j'], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("Emergency cleanup completed successfully")
                return self.find_sas_processes().__len__()
            else:
                logger.error(f"Emergency cleanup failed: {result.stderr}")
                return 0
                
        except Exception as e:
            logger.error(f"Emergency cleanup error: {e}")
            return 0


def main():
    """Main monitoring function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    monitor = SASSessionMonitor()
    
    # Get current stats
    stats = monitor.get_session_stats()
    logger.info(f"Current SAS sessions: {stats['total_sessions']}")
    logger.info(f"Total CPU usage: {stats['total_cpu_percent']:.1f}%")
    logger.info(f"Total memory usage: {stats['total_mem_percent']:.1f}%")
    
    # Clean up old sessions
    cleaned = monitor.cleanup_old_sessions()
    if cleaned > 0:
        logger.info(f"Cleaned up {cleaned} old SAS sessions")
    
    # Show remaining processes
    remaining = monitor.get_session_stats()
    logger.info(f"Remaining SAS sessions: {remaining['total_sessions']}")


if __name__ == "__main__":
    main() 