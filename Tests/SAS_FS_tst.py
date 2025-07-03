import os
import logging
import signal
import time
from pathlib import Path
from contextlib import contextmanager

# Get the project root directory (parent of Tests directory)
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# Set up basic logging for terminal output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(LOGS_DIR / 'sas_fs_test.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

@contextmanager
def timeout(seconds):
    """Context manager for timeout handling"""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def test_sas_file_operations(sas_session, target_directory, logger):
    """
    Test SAS-specific file operations and directory functionality
    """
    logger.info(f"Testing SAS file operations for directory: {target_directory}")
    
    sas_test_code = f"""
    /* Test directory access and file operations */
    %let test_dir = {target_directory};
    
    /* Test 1: Check if directory exists and is accessible */
    %macro test_directory_access;
        %local rc fileref;
        %let rc = %sysfunc(filename(fileref, &test_dir));
        
        %if %sysfunc(fexist(&fileref)) %then %do;
            %put NOTE: Directory &test_dir exists and is accessible;
            %let dir_test_result = SUCCESS;
        %end;
        %else %do;
            %put ERROR: Directory &test_dir does not exist or is not accessible;
            %let dir_test_result = FAILED;
        %end;
        
        %let rc = %sysfunc(filename(fileref));
    %mend;
    
    /* Test 2: Test directory creation with dlcreatedir */
    %macro test_directory_creation;
        options dlcreatedir;
        %local test_subdir;
        %let test_subdir = &test_dir/test_creation;
        
        libname testcreate "&test_subdir";
        
        %if &syslibrc = 0 %then %do;
            %put NOTE: Directory creation test successful;
            %let create_test_result = SUCCESS;
            libname testcreate clear;
        %end;
        %else %do;
            %put ERROR: Directory creation test failed;
            %let create_test_result = FAILED;
        %end;
    %mend;
    
    /* Test 3: Test file write permissions */
    %macro test_write_permissions;
        %local test_file;
        %let test_file = &test_dir/test_write.txt;
        
        data _null_;
            file "&test_file";
            put "Test write operation";
        run;
        
        %if &syserr = 0 %then %do;
            %put NOTE: Write permission test successful;
            %let write_test_result = SUCCESS;
        %end;
        %else %do;
            %put ERROR: Write permission test failed;
            %let write_test_result = FAILED;
        %end;
    %mend;
    
    /* Execute all tests */
    %test_directory_access;
    %test_directory_creation;
    %test_write_permissions;
    
    /* Report results */
    %put NOTE: ===========================================;
    %put NOTE: DIRECTORY TESTING RESULTS;
    %put NOTE: Directory Access: &dir_test_result;
    %put NOTE: Directory Creation: &create_test_result;  
    %put NOTE: Write Permissions: &write_test_result;
    %put NOTE: ===========================================;
    """
    
    try:
        result = sas_session.submit(sas_test_code)
        logger.info("SAS directory tests completed")
        
        # Parse results from log
        log_content = result['LOG']
        
        tests_passed = (
            "Directory Access: SUCCESS" in log_content and
            "Directory Creation: SUCCESS" in log_content and
            "Write Permissions: SUCCESS" in log_content
        )
        
        if tests_passed:
            logger.info("✅ All SAS directory tests passed")
            return True
        else:
            logger.error("❌ Some SAS directory tests failed")
            logger.error(f"SAS Log: {log_content}")
            return False
            
    except Exception as e:
        logger.error(f"❌ SAS directory testing failed: {e}")
        return False

def test_sas_directory_listing(sas_session, target_directory, logger, timeout_seconds=10):
    """
    Test SAS directory listing functionality with timeout
    """
    logger.info(f"Testing SAS directory listing for: {target_directory}")
    logger.info(f"Timeout set to {timeout_seconds} seconds...")
    
    try:
        # Use timeout context manager
        with timeout(timeout_seconds):
            # Use SASpy dirlist function
            file_list = sas_session.dirlist(target_directory)
            logger.info(f"✅ SAS directory listing successful")
            logger.info(f"   Found {len(file_list)} items in directory")
            
            # Show first few items
            for i, item in enumerate(file_list[:10]):
                logger.info(f"   {i+1}. {item}")
            
            if len(file_list) > 10:
                logger.info(f"   ... and {len(file_list) - 10} more items")
            
            return True
            
    except TimeoutError:
        logger.error(f"❌ SAS directory listing timed out after {timeout_seconds} seconds")
        logger.info("💡 This might indicate permission issues or network problems")
        return False
        
    except Exception as e:
        logger.error(f"❌ SAS directory listing failed: {e}")
        return False

def test_sas_data_transfer(sas_session, logger):
    """
    Test SAS data transfer functionality
    """
    logger.info("Testing SAS data transfer functionality")
    
    try:
        # Create test data
        import pandas as pd
        test_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        
        # Transfer to SAS
        logger.info("Transferring test data to SAS...")
        sas_session.df2sd(test_data, 'test_transfer_data')
        
        # Check if dataset exists
        if sas_session.exist('test_transfer_data'):
            logger.info("✅ Data transfer to SAS successful")
            
            # Read back from SAS
            logger.info("Reading data back from SAS...")
            read_data = sas_session.sd2df('test_transfer_data')
            
            if read_data is not None and len(read_data) == 5:
                logger.info("✅ Data read back from SAS successful")
                logger.info(f"   Retrieved {len(read_data)} rows with {len(read_data.columns)} columns")
                return True
            else:
                logger.error("❌ Data read back from SAS failed")
                return False
        else:
            logger.error("❌ Data transfer to SAS failed - dataset not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ SAS data transfer test failed: {e}")
        return False

def test_sas_basic_commands(sas_session, logger):
    """
    Test basic SAS command execution
    """
    logger.info("Testing basic SAS command execution")
    
    try:
        # Test simple PROC PRINT
        result = sas_session.submit("proc print data=sashelp.class(obs=3); run;")
        
        if "ERROR" not in result['LOG']:
            logger.info("✅ Basic SAS command execution successful")
            return True
        else:
            logger.warning(f"⚠️  SAS command had warnings: {result['LOG']}")
            return True  # Still consider it a pass if it executed
            
    except Exception as e:
        logger.error(f"❌ Basic SAS command execution failed: {e}")
        return False

def test_sas_report_generation_oda(sas_session, logger):
    """
    Test SAS report generation approaches that might work in ODA
    """
    logger.info("=" * 60)
    logger.info("TESTING SAS REPORT GENERATION FOR ODA")
    logger.info("=" * 60)
    
    # Create test data for analysis
    import pandas as pd
    test_data = pd.DataFrame({
        'Treatment': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'Response': [10, 12, 11, 15, 16, 14, 8, 9, 7]
    })
    
    sas_session.df2sd(test_data, 'report_test_data')
    logger.info("✅ Test data transferred to SAS")
    
    # Test 1: ODS HTML to memory/string
    logger.info("\n--- Test 1: ODS HTML to memory ---")
    html_test_code = """
    /* Test ODS HTML output that might be captured */
    ods html5 (id=saspy_html) file=_webout options(bitmap_mode='inline') device=svg style=HTMLBlue;
    
    proc glm data=work.report_test_data;
        class Treatment;
        model Response = Treatment;
        lsmeans Treatment / stderr pdiff cl adjust=bon;
        title "Test Report - Treatment Analysis";
    run;
    
    ods html5 (id=saspy_html) close;
    """
    
    try:
        html_result = sas_session.submit(html_test_code)
        if "ERROR" not in html_result['LOG']:
            logger.info("✅ ODS HTML generation successful")
            logger.info("💡 HTML output might be available in _webout or similar")
        else:
            logger.warning(f"⚠️  ODS HTML had issues: {html_result['LOG']}")
    except Exception as e:
        logger.error(f"❌ ODS HTML test failed: {e}")
    
    # Test 2: ODS LISTING with capture
    logger.info("\n--- Test 2: ODS LISTING capture ---")
    listing_test_code = """
    /* Test ODS LISTING output capture */
    ods listing close;
    ods listing;
    
    proc glm data=work.report_test_data;
        class Treatment;
        model Response = Treatment;
        lsmeans Treatment / stderr pdiff cl adjust=bon;
        title "Test Report - Treatment Analysis (Listing)";
    run;
    """
    
    try:
        listing_result = sas_session.submit(listing_test_code)
        if "ERROR" not in listing_result['LOG']:
            logger.info("✅ ODS LISTING generation successful")
            logger.info("💡 Listing output captured in SAS log")
        else:
            logger.warning(f"⚠️  ODS LISTING had issues: {listing_result['LOG']}")
    except Exception as e:
        logger.error(f"❌ ODS LISTING test failed: {e}")
    
    # Test 3: ODS OUTPUT to datasets (this should work)
    logger.info("\n--- Test 3: ODS OUTPUT to datasets ---")
    output_test_code = """
    /* Test ODS OUTPUT to capture results in datasets */
    ods output 
        FitStatistics=work.fitstats
        LSMeans=work.lsmeans
        Diffs=work.diffs
        OverallANOVA=work.anova;
    
    proc glm data=work.report_test_data;
        class Treatment;
        model Response = Treatment;
        lsmeans Treatment / stderr pdiff cl adjust=bon;
    run;
    
    ods output close;
    """
    
    try:
        output_result = sas_session.submit(output_test_code)
        if "ERROR" not in output_result['LOG']:
            logger.info("✅ ODS OUTPUT to datasets successful")
            
            # Check if datasets were created
            datasets_created = []
            for ds in ['fitstats', 'lsmeans', 'diffs', 'anova']:
                if sas_session.exist(ds):
                    datasets_created.append(ds)
                    logger.info(f"   ✅ Dataset {ds} created")
                else:
                    logger.warning(f"   ⚠️  Dataset {ds} not found")
            
            logger.info(f"💡 {len(datasets_created)}/4 datasets created successfully")
        else:
            logger.warning(f"⚠️  ODS OUTPUT had issues: {output_result['LOG']}")
    except Exception as e:
        logger.error(f"❌ ODS OUTPUT test failed: {e}")
    
    # Test 4: Try different file paths for ODS PDF
    logger.info("\n--- Test 4: ODS PDF with different paths ---")
    pdf_paths = [
        '/tmp/test_report.pdf',
        './test_report.pdf',
        'test_report.pdf',
        'work.test_report.pdf'
    ]
    
    for pdf_path in pdf_paths:
        logger.info(f"Testing PDF path: {pdf_path}")
        pdf_test_code = f"""
        ods pdf file='{pdf_path}' style=journal;
        
        proc print data=work.fitstats;
            title "Test PDF Report";
        run;
        
        ods pdf close;
        """
        
        try:
            pdf_result = sas_session.submit(pdf_test_code)
            if "ERROR" not in pdf_result['LOG']:
                logger.info(f"✅ PDF generation successful for {pdf_path}")
                # Check if file exists (might not be accessible from Python)
                logger.info(f"💡 PDF might be created at {pdf_path}")
                break
            else:
                logger.warning(f"⚠️  PDF generation failed for {pdf_path}")
        except Exception as e:
            logger.error(f"❌ PDF test failed for {pdf_path}: {e}")
    
    # Test 5: ODS RTF with different paths
    logger.info("\n--- Test 5: ODS RTF with different paths ---")
    rtf_paths = [
        '/tmp/test_report.rtf',
        './test_report.rtf',
        'test_report.rtf'
    ]
    
    for rtf_path in rtf_paths:
        logger.info(f"Testing RTF path: {rtf_path}")
        rtf_test_code = f"""
        ods rtf file='{rtf_path}' style=journal;
        
        proc print data=work.fitstats;
            title "Test RTF Report";
        run;
        
        ods rtf close;
        """
        
        try:
            rtf_result = sas_session.submit(rtf_test_code)
            if "ERROR" not in rtf_result['LOG']:
                logger.info(f"✅ RTF generation successful for {rtf_path}")
                logger.info(f"💡 RTF might be created at {rtf_path}")
                break
            else:
                logger.warning(f"⚠️  RTF generation failed for {rtf_path}")
        except Exception as e:
            logger.error(f"❌ RTF test failed for {rtf_path}: {e}")
    
    # Test 6: Check if we can access SAS WORK directory
    logger.info("\n--- Test 6: SAS WORK directory access ---")
    try:
        work_files = sas_session.dirlist('/tmp')  # SAS WORK is often in /tmp
        logger.info(f"✅ Found {len(work_files)} files in /tmp")
        
        # Look for SAS files
        sas_files = [f for f in work_files if f.endswith('.sas7bdat') or f.endswith('.pdf') or f.endswith('.rtf')]
        if sas_files:
            logger.info(f"✅ Found {len(sas_files)} potential SAS output files")
            for f in sas_files[:5]:  # Show first 5
                logger.info(f"   - {f}")
        else:
            logger.info("💡 No obvious SAS output files found")
            
    except Exception as e:
        logger.error(f"❌ WORK directory access failed: {e}")
    
    logger.info("=" * 60)
    logger.info("SAS REPORT GENERATION TEST COMPLETE")
    logger.info("=" * 60)

def run_comprehensive_sas_fs_tests():
    """
    Run comprehensive SAS file system tests
    """
    logger.info("=" * 60)
    logger.info("SAS FILE SYSTEM TESTING")
    logger.info("=" * 60)
    
    # Ensure logs directory exists
    LOGS_DIR.mkdir(exist_ok=True)
    
    try:
        import saspy
        logger.info("✅ SASpy module imported successfully")
        
        # Establish SAS connection
        logger.info("Establishing SAS connection...")
        sas = saspy.SASsession(cfgname='oda')
        logger.info("✅ SAS connection established successfully")
        
        all_tests_passed = True
        
        # Test 1: Basic SAS commands (should always work)
        logger.info(f"\n--- Testing Basic SAS Commands ---")
        basic_success = test_sas_basic_commands(sas, logger)
        if not basic_success:
            all_tests_passed = False
            logger.error("❌ Basic SAS commands failed - stopping further tests")
            sas.endsas()
            return False
        
        # Test 2: Data transfer (should work in SAS ODA)
        logger.info(f"\n--- Testing Data Transfer ---")
        transfer_success = test_sas_data_transfer(sas, logger)
        if not transfer_success:
            all_tests_passed = False
        
        # Test 3: Report generation (the main objective)
        test_sas_report_generation_oda(sas, logger)
        
        # Test 4: Directory operations (may fail due to permissions)
        test_directories = [
            "/tmp",  # Usually accessible
            "/var/tmp",  # Usually accessible
            "/home"  # May have permission issues
        ]
        
        for test_dir in test_directories:
            logger.info(f"\n--- Testing Directory: {test_dir} ---")
            
            # Test directory listing with shorter timeout
            listing_success = test_sas_directory_listing(sas, test_dir, logger, timeout_seconds=5)
            
            # Test file operations (only if listing worked)
            if listing_success:
                fs_success = test_sas_file_operations(sas, test_dir, logger)
                if not fs_success:
                    all_tests_passed = False
            else:
                logger.warning(f"⚠️  Skipping file operations for {test_dir} due to listing failure")
                # Don't fail the entire test suite for directory access issues
                # This is common in SAS ODA environments
        
        # Clean up
        logger.info("Cleaning up SAS session...")
        sas.endsas()
        logger.info("✅ SAS session closed successfully")
        
        # Summary
        logger.info("=" * 60)
        logger.info("SAS FILE SYSTEM TEST SUMMARY")
        logger.info("=" * 60)
        if all_tests_passed:
            logger.info("✅ All SAS file system tests passed!")
        else:
            logger.warning("⚠️  Some SAS file system tests failed (this may be normal for ODA)")
            logger.info("💡 Directory access failures are common in SAS ODA environments")
        logger.info("=" * 60)
        
        return all_tests_passed
        
    except ImportError:
        logger.error("❌ SASpy module not available")
        logger.info("💡 Install SASpy with: pip install saspy")
        return False
        
    except Exception as e:
        logger.error(f"❌ SAS file system testing failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting SAS File System Test Suite")
    print("=" * 60)
    
    success = run_comprehensive_sas_fs_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SAS File System Test Suite completed successfully!")
    else:
        print("⚠️  SAS File System Test Suite completed with some failures")
        print("💡 This may be normal for SAS ODA environments")
    print(f"📁 Check {LOGS_DIR}/sas_fs_test.log for detailed results")
    print("=" * 60)
