import os
import pandas as pd
import logging
from pathlib import Path

# Get the project root directory (parent of Tests directory)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Set up basic logging for terminal output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(LOGS_DIR / 'sas_test.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

def test_sas_directory_access(sas_session, base_path, logger):
    """
    Test directory access and file operations for SAS ODA environment
    """
    try:
        # Test 1: Verify base directory path format
        if not base_path.startswith(('~/', '/home/')):
            logger.warning(f"Path may not be valid for SAS ODA: {base_path}")
            return False
            
        # Test 2: Use SASpy dirlist to check directory accessibility
        try:
            file_list = sas_session.dirlist(base_path)
            logger.info(f"Successfully accessed directory: {base_path}")
            logger.info(f"Found {len(file_list)} items in directory")
            
            # Separate files from directories
            files = [f for f in file_list if not f.endswith('/')]
            dirs = [d for d in file_list if d.endswith('/')]
            
            logger.info(f"Files: {len(files)}, Directories: {len(dirs)}")
            
        except Exception as e:
            logger.error(f"Cannot access SAS directory {base_path}: {e}")
            return False
            
        # Test 3: Verify directory creation capability
        test_dir = f"{base_path}/test_directory"
        try:
            sas_code = f"""
            options dlcreatedir;
            libname testlib "{test_dir}";
            libname testlib clear;
            """
            result = sas_session.submit(sas_code)
            
            if "ERROR" in result['LOG']:
                logger.error(f"Directory creation test failed: {result['LOG']}")
                return False
            else:
                logger.info("Directory creation test passed")
                
        except Exception as e:
            logger.error(f"Directory creation test failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Directory testing failed: {e}")
        return False

def validate_data_file_for_sas(data_file_path, required_columns, logger):
    """
    Validate data file before processing in SAS
    Similar to your pandas validation but SAS-focused
    """
    try:
        # Test file existence and readability
        if not os.path.exists(data_file_path):
            logger.error(f"Data file does not exist: {data_file_path}")
            return None
            
        if not os.access(data_file_path, os.R_OK):
            logger.error(f"No read permission for file: {data_file_path}")
            return None
            
        # Load and validate data structure
        try:
            if data_file_path.endswith('.csv'):
                data = pd.read_csv(data_file_path)
            elif data_file_path.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(data_file_path)
            elif data_file_path.endswith('.sas7bdat'):
                data = pd.read_sas(data_file_path)
            else:
                logger.error(f"Unsupported file format: {data_file_path}")
                return None
                
            logger.info(f"Loaded  {data.shape}")
            
            # Validate required columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None
                
            # Check for empty dataset
            if data.empty:
                logger.error("Dataset is empty")
                return None
                
            # Return data info for further processing
            data_info = {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'memory_usage': data.memory_usage(deep=True).sum(),
                'has_nulls': data.isnull().any().any()
            }
            
            logger.info(f"Data validation successful: {data_info}")
            return data_info
            
        except Exception as e:
            logger.error(f"Failed to load/validate data file: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Data file validation failed: {e}")
        return None

def test_local_file_system():
    """
    Test local file system functionality without SAS
    """
    logger.info("=" * 60)
    logger.info("TESTING LOCAL FILE SYSTEM FUNCTIONALITY")
    logger.info("=" * 60)
    
    # Ensure logs directory exists
    LOGS_DIR.mkdir(exist_ok=True)
    logger.info(f"✅ Created logs directory: {LOGS_DIR}")
    
    # Test current working directory
    cwd = os.getcwd()
    logger.info(f"✅ Current working directory: {cwd}")
    logger.info(f"✅ Project root directory: {PROJECT_ROOT}")
    
    # Test data directory access
    if DATA_DIR.exists():
        logger.info(f"✅ Data directory exists: {DATA_DIR}")
        data_files = list(DATA_DIR.iterdir())
        logger.info(f"✅ Found {len(data_files)} files in data directory")
        for file in data_files:
            logger.info(f"   - {file.name}")
    else:
        logger.warning(f"⚠️  Data directory not found: {DATA_DIR}")
    
    # Test data file validation
    test_files = [
        (DATA_DIR / "simple_example.csv", ["Treatment", "TumorSize"]),
        (DATA_DIR / "repeated_example.csv", ["Treatment", "TumorSize", "Week", "Dog"])
    ]
    
    for file_path, required_columns in test_files:
        logger.info(f"Testing file: {file_path}")
        result = validate_data_file_for_sas(str(file_path), required_columns, logger)
        if result:
            logger.info(f"✅ File validation successful for {file_path.name}")
        else:
            logger.error(f"❌ File validation failed for {file_path.name}")
    
    # Test file creation and writing
    test_file = LOGS_DIR / "test_write.txt"
    try:
        with open(test_file, 'w') as f:
            f.write("Test write operation\n")
        logger.info(f"✅ Successfully wrote to {test_file}")
        
        # Test file reading
        with open(test_file, 'r') as f:
            content = f.read()
        logger.info(f"✅ Successfully read from {test_file}: {content.strip()}")
        
    except Exception as e:
        logger.error(f"❌ File write/read test failed: {e}")
    
    # Test other important directories
    important_dirs = [
        PROJECT_ROOT / "utils",
        PROJECT_ROOT / "outputs", 
        PROJECT_ROOT / "plots"
    ]
    
    logger.info("Testing important project directories:")
    for dir_path in important_dirs:
        if dir_path.exists():
            logger.info(f"✅ Directory exists: {dir_path.name}")
        else:
            logger.warning(f"⚠️  Directory not found: {dir_path.name}")
    
    # Test main application files
    main_files = [
        PROJECT_ROOT / "RMND_V1.py",
        PROJECT_ROOT / "sgsnm_v3.py",
        PROJECT_ROOT / "master_app.py"
    ]
    
    logger.info("Testing main application files:")
    for file_path in main_files:
        if file_path.exists():
            logger.info(f"✅ File exists: {file_path.name}")
        else:
            logger.warning(f"⚠️  File not found: {file_path.name}")
    
    logger.info("=" * 60)
    logger.info("LOCAL FILE SYSTEM TEST COMPLETE")
    logger.info("=" * 60)

def test_sas_connectivity():
    """
    Test SAS connectivity if available
    """
    logger.info("=" * 60)
    logger.info("TESTING SAS CONNECTIVITY")
    logger.info("=" * 60)
    
    try:
        import saspy
        logger.info("✅ SASpy module imported successfully")
        
        # Try to establish SAS connection
        try:
            sas = saspy.SASsession(cfgname='oda')
            logger.info("✅ SAS connection established successfully")
            
            # Test basic SAS functionality
            result = sas.submit("proc print data=sashelp.class(obs=1); run;")
            if "ERROR" not in result['LOG']:
                logger.info("✅ Basic SAS command execution successful")
            else:
                logger.warning(f"⚠️  SAS command had warnings: {result['LOG']}")
            
            # Test directory access (if we have a valid path)
            test_path = "/home"
            try:
                file_list = sas.dirlist(test_path)
                logger.info(f"✅ SAS directory access successful for {test_path}")
                logger.info(f"   Found {len(file_list)} items")
            except Exception as e:
                logger.warning(f"⚠️  SAS directory access failed for {test_path}: {e}")
            
            # Clean up
            sas.endsas()
            logger.info("✅ SAS session closed successfully")
            
        except Exception as e:
            logger.error(f"❌ SAS connection failed: {e}")
            logger.info("💡 This is expected if SAS is not configured or not available")
    
    except ImportError:
        logger.warning("⚠️  SASpy module not available - skipping SAS tests")
        logger.info("💡 Install SASpy with: pip install saspy")
    
    logger.info("=" * 60)
    logger.info("SAS CONNECTIVITY TEST COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    print("🚀 Starting SAS Directory Test Suite")
    print("=" * 60)
    
    # Run local file system tests
    test_local_file_system()
    
    # Run SAS connectivity tests
    test_sas_connectivity()
    
    print("\n" + "=" * 60)
    print("✅ Test suite completed!")
    print(f"📁 Check {LOGS_DIR}/sas_test.log for detailed results")
    print("=" * 60)
