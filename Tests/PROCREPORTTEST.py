import saspy
import pandas as pd
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create log directory
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'procreport_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Add file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("=" * 60)
logger.info("PROC REPORT TEST - STARTING ANALYSIS")
logger.info("=" * 60)

try:
    # Step 1: Load data
    logger.info("Step 1: Loading data from CSV file")
    data_file = 'data/simple_example.csv'
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    data = pd.read_csv(data_file)
    logger.info(f"Data loaded successfully: {data.shape}")
    logger.info(f"Columns: {list(data.columns)}")
    logger.info(f"Data types: {data.dtypes.to_dict()}")
    logger.info(f"First few rows:")
    logger.info(data.head().to_string())
    
    # Step 2: Establish SAS connection
    logger.info("Step 2: Establishing SAS connection")
    sas = saspy.SASsession()
    logger.info("SAS connection established successfully")
    
    # Step 3: Transfer data to SAS
    logger.info("Step 3: Transferring data to SAS")
    logger.info("Converting Treatment column to string for SAS compatibility")
    if 'Treatment' in data.columns:
        data['Treatment'] = data['Treatment'].astype(str)
    
    sas.df2sd(data, 'testdata')
    logger.info("Data transferred to SAS table 'work.testdata'")
    
    # Step 4: Verify data in SAS
    logger.info("Step 4: Verifying data in SAS")
    verify_result = sas.submit("""
    proc contents data=work.testdata;
    run;
    proc print data=work.testdata(obs=5);
    run;
    """)
    logger.info("SAS data verification log:")
    logger.info(verify_result['LOG'])
    
    # Step 5: Run main analysis
    logger.info("Step 5: Running main PROC GLM analysis")
    sas_code = """
    /* Enable ODS TRACE to see available tables */
    ods trace on;
    
    /* Capture ODS tables */
    ods output FitStatistics=work.fitstats Diff=work.diffs;
    
    /* Run PROC GLM */
    proc glm data=work.testdata;
        class Treatment;
        model TumorSize = Treatment;
        lsmeans Treatment / stderr pdiff cl adjust=bon;
    run;
    
    /* Verify datasets were created */
    proc contents data=work.fitstats; 
    run;
    proc contents data=work.diffs; 
    run;
    
    /* Print datasets for verification */
    proc print data=work.fitstats; 
    run;
    proc print data=work.diffs; 
    run;
    
    /* Show ODS trace information */
    ods trace off;
    """
    
    result = sas.submit(sas_code)
    logger.info("Main analysis completed")
    logger.info("SAS execution log:")
    logger.info(result['LOG'])
    
    # Step 6: Check for errors and warnings
    logger.info("Step 6: Analyzing SAS log for errors and warnings")
    log_content = result['LOG']
    
    # Check for errors
    error_count = log_content.count('ERROR:')
    warning_count = log_content.count('WARNING:')
    note_count = log_content.count('NOTE:')
    
    logger.info(f"Log analysis summary:")
    logger.info(f"  - Errors found: {error_count}")
    logger.info(f"  - Warnings found: {warning_count}")
    logger.info(f"  - Notes found: {note_count}")
    
    if error_count > 0:
        logger.error("ERRORS DETECTED IN SAS LOG:")
        lines = log_content.split('\n')
        for i, line in enumerate(lines):
            if 'ERROR:' in line:
                logger.error(f"  Line {i+1}: {line.strip()}")
    
    if warning_count > 0:
        logger.warning("WARNINGS DETECTED IN SAS LOG:")
        lines = log_content.split('\n')
        for i, line in enumerate(lines):
            if 'WARNING:' in line:
                logger.warning(f"  Line {i+1}: {line.strip()}")
    
    # Step 7: Verify datasets exist in Python
    logger.info("Step 7: Verifying datasets can be extracted to Python")
    try:
        if sas.exist('fitstats'):
            fitstats_df = sas.sd2df('fitstats')
            logger.info(f"Fitstats dataset extracted: {fitstats_df.shape}")
            logger.info(f"Fitstats columns: {list(fitstats_df.columns)}")
            logger.info("Fitstats data:")
            logger.info(fitstats_df.to_string())
        else:
            logger.error("Fitstats dataset not found in SAS")
    except Exception as e:
        logger.error(f"Failed to extract fitstats dataset: {e}")
    
    try:
        if sas.exist('diffs'):
            diffs_df = sas.sd2df('diffs')
            logger.info(f"Diffs dataset extracted: {diffs_df.shape}")
            logger.info(f"Diffs columns: {list(diffs_df.columns)}")
            logger.info("Diffs data:")
            logger.info(diffs_df.to_string())
        else:
            logger.error("Diffs dataset not found in SAS")
    except Exception as e:
        logger.error(f"Failed to extract diffs dataset: {e}")
    
    # Step 8: Test ODS output (if you want to test PDF/RTF generation)
    logger.info("Step 8: Testing ODS PDF/RTF output")
    test_ods_code = """
    /* Test ODS output to a simple location */
    ods pdf file='/tmp/test_output.pdf' style=journal;
    ods rtf file='/tmp/test_output.rtf' style=journal;
    
    proc print data=work.fitstats;
        title "Test ODS Output - Fit Statistics";
    run;
    
    proc print data=work.diffs;
        title "Test ODS Output - Pairwise Differences";
    run;
    
    ods pdf close;
    ods rtf close;
    """
    
    ods_result = sas.submit(test_ods_code)
    logger.info("ODS test completed")
    logger.info("ODS test log:")
    logger.info(ods_result['LOG'])
    
    # Check if test files were created
    if os.path.exists('/tmp/test_output.pdf'):
        logger.info("✅ Test PDF file created successfully")
    else:
        logger.warning("❌ Test PDF file not found")
    
    if os.path.exists('/tmp/test_output.rtf'):
        logger.info("✅ Test RTF file created successfully")
    else:
        logger.warning("❌ Test RTF file not found")
    
    logger.info("=" * 60)
    logger.info("PROC REPORT TEST - COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
except Exception as e:
    logger.error(f"Test failed with exception: {e}")
    logger.error(f"Exception type: {type(e).__name__}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    
finally:
    # Clean up
    try:
        if 'sas' in locals():
            logger.info("Cleaning up SAS session")
            sas.endsas()
            logger.info("SAS session closed")
    except Exception as e:
        logger.error(f"Failed to close SAS session: {e}")
    
    logger.info(f"Log file saved to: {log_file}")
    print(f"\nTest completed. Check log file: {log_file}")