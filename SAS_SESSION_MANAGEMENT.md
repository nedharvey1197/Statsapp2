# SAS Session Management Guide

This guide explains how to use the SAS session management utilities to prevent orphaned sessions and maintain optimal system performance.

## 🚨 Problem Solved

**Issue**: Orphaned SAS Java processes consuming 90%+ CPU and causing system slowdowns.

**Root Cause**: 
- Double SAS session creation in applications
- Inconsistent cleanup methods (`sas.end()` vs `sas.endsas()`)
- No session monitoring or cleanup mechanisms
- Multiple Streamlit applications running simultaneously

**Solution**: Integrated session management with monitoring and automatic cleanup.

## 🛠️ Utilities Created

### 1. Session Monitor (`utils/sas_session_monitor.py`)

**Purpose**: Monitors and manages SAS Java processes to prevent resource leaks.

**Features**:
- Find all running SAS processes
- Clean up old sessions (configurable age threshold)
- Emergency cleanup of all SAS processes
- Session statistics (CPU, memory usage)

**Usage**:
```bash
# Check current session status
python utils/sas_session_monitor.py

# Clean up sessions older than 1 hour (default)
python utils/sas_session_monitor.py

# Emergency cleanup (kills all SAS processes)
python -c "from utils.sas_session_monitor import SASSessionMonitor; SASSessionMonitor().emergency_cleanup()"
```

### 2. Startup Script (`start_sas_app.py`)

**Purpose**: Ensures clean SAS environment before starting applications.

**Features**:
- Automatic cleanup of existing SAS sessions
- Port availability checking
- Session monitoring integration
- Direct Streamlit app launching

**Usage**:
```bash
# Clean environment and start app
python start_sas_app.py sgsnm_v3.py

# Check session status only
python start_sas_app.py --monitor-only

# Skip cleanup step
python start_sas_app.py sgsnm_v3.py --no-cleanup

# Use different port
python start_sas_app.py sgsnm_v3.py --port 8502
```

### 3. Enhanced SAS Analysis Manager (`utils/sas_analysis_manager.py`)

**Purpose**: Centralized SAS connection management with session monitoring.

**New Features**:
- Automatic environment cleanup before connections
- Session monitoring integration
- Enhanced error handling and recovery
- Session statistics tracking

## 🔧 Integration Points

### 1. In `sgsnm_v3.py` (Main Application)

**Changes Made**:
- Removed duplicate SAS session creation
- Added session monitoring to sidebar
- Pre-analysis cleanup
- Real-time session statistics

**New Features**:
```python
# Session monitoring in sidebar
with st.sidebar:
    st.subheader("🔍 SAS Session Monitor")
    stats = session_monitor.get_session_stats()
    st.metric("Active Sessions", stats['total_sessions'])
    st.metric("CPU Usage", f"{stats['total_cpu_percent']:.1f}%")
    st.metric("Memory Usage", f"{stats['total_mem_percent']:.1f}%")
    
    if st.button("🧹 Clean Old Sessions"):
        cleaned = session_monitor.cleanup_old_sessions()
        st.success(f"Cleaned {cleaned} old sessions")
```

### 2. In SAS Connection Manager

**Enhanced Features**:
```python
# Automatic environment cleanup
def connect(self) -> Optional[saspy.SASsession]:
    self.cleanup_environment()  # Clean before connecting
    # ... connection logic

# Session statistics
def get_session_stats(self) -> Dict[str, Any]:
    if self.session_monitor:
        return self.session_monitor.get_session_stats()
```

## 📋 Best Practices

### 1. Before Starting SAS Work

```bash
# Option 1: Use startup script
python start_sas_app.py your_app.py

# Option 2: Manual cleanup
python utils/sas_session_monitor.py
```

### 2. During Development

```python
# In your Streamlit app
if SESSION_MONITOR_AVAILABLE:
    # Monitor sessions in sidebar
    stats = session_monitor.get_session_stats()
    # Display metrics and cleanup button
```

### 3. After Analysis

```python
# The manager automatically handles cleanup
# But you can also manually check
python utils/sas_session_monitor.py
```

### 4. Emergency Situations

```bash
# Kill all SAS processes
pkill -f saspy2j

# Or use the monitor
python -c "from utils.sas_session_monitor import SASSessionMonitor; SASSessionMonitor().emergency_cleanup()"
```

## 🔍 Monitoring and Debugging

### 1. Check Current Status

```bash
# Quick status check
python utils/sas_session_monitor.py

# Detailed process list
ps aux | grep saspy2j | grep -v grep

# Network connections
lsof -i | grep sas
```

### 2. Performance Monitoring

```bash
# System performance
top -l 1 -n 10 -o cpu

# Memory usage
vm_stat

# Process details
ps aux | grep java | head -5
```

### 3. Log Analysis

```bash
# Check application logs
tail -f logs/your_app.log

# Check SAS session monitor logs
python utils/sas_session_monitor.py 2>&1 | tee session_monitor.log
```

## 🚀 Quick Start Guide

### 1. First Time Setup

```bash
# 1. Clean any existing sessions
python start_sas_app.py --monitor-only

# 2. Start your application
python start_sas_app.py sgsnm_v3.py
```

### 2. Daily Usage

```bash
# Start with clean environment
python start_sas_app.py sgsnm_v3.py

# Monitor during use (via Streamlit sidebar)
# Clean old sessions as needed
```

### 3. Troubleshooting

```bash
# If system is slow
python utils/sas_session_monitor.py

# If port is in use
python start_sas_app.py --port 8502

# Emergency cleanup
pkill -f saspy2j
```

## 📊 Expected Results

### Before (Problem State):
- **20+ SAS Java processes** running for 24+ hours
- **90%+ CPU usage** from orphaned sessions
- **High memory pressure** and swap activity
- **System slowdowns** and unresponsiveness

### After (Solution State):
- **0-2 active SAS sessions** (only when needed)
- **Normal CPU usage** (5-20% during analysis)
- **Clean memory usage** with no swap pressure
- **Responsive system** performance

## 🔧 Configuration Options

### Session Monitor Settings

```python
# In utils/sas_session_monitor.py
class SASSessionMonitor:
    def __init__(self):
        self.max_session_age = 3600  # 1 hour default
        # Adjust based on your needs
```

### Connection Manager Settings

```python
# In utils/sas_analysis_manager.py
class SASConnectionManager:
    def __init__(self, config_name: str = 'oda'):
        self.max_retries = 3  # Connection retry attempts
        # Adjust based on your SAS environment
```

## 🆘 Troubleshooting

### Common Issues

1. **"Session monitor not available"**
   - Check that `utils/sas_session_monitor.py` exists
   - Verify Python path includes utils directory

2. **"Permission denied" killing processes**
   - Run with appropriate permissions
   - Use `sudo` if necessary (rare)

3. **"Port already in use"**
   - Use `--port` option to specify different port
   - Or let startup script handle it automatically

4. **"SAS connection failed"**
   - Check SAS configuration
   - Verify network connectivity to SAS servers
   - Check SAS ODA credentials

### Emergency Procedures

```bash
# 1. Stop all SAS processes
pkill -f saspy2j

# 2. Check for remaining processes
ps aux | grep saspy2j

# 3. Restart with clean environment
python start_sas_app.py your_app.py
```

## 📈 Performance Metrics

### Monitoring Dashboard

The Streamlit sidebar now includes:
- **Active Sessions**: Number of current SAS processes
- **CPU Usage**: Total CPU consumption by SAS processes
- **Memory Usage**: Total memory consumption by SAS processes
- **Cleanup Button**: Manual cleanup of old sessions

### Expected Values

- **Active Sessions**: 0-2 (0 when idle, 1-2 during analysis)
- **CPU Usage**: 0-20% (0% when idle, 5-20% during analysis)
- **Memory Usage**: 0-5% (0% when idle, 1-5% during analysis)

## 🎯 Success Criteria

✅ **No orphaned SAS processes** after analysis completion  
✅ **System performance** remains responsive  
✅ **Memory usage** stays within normal ranges  
✅ **CPU usage** returns to baseline after analysis  
✅ **Automatic cleanup** prevents session accumulation  
✅ **Real-time monitoring** provides visibility into session status  

---

**Note**: These utilities are designed to work with SAS ODA (OnDemand for Academics) and saspy. Adjustments may be needed for other SAS environments. 