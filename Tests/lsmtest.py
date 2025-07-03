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
log_file = os.path.join(log_dir, f'lsmtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Add file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("=" * 60)
logger.info("LSMEANS TEST - STARTING ANALYSIS")
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
    
    # Step 5: Run LSMeans analysis
    logger.info("Step 5: Running LSMeans analysis")
    sas_code = """
    /* Set error handling */
    options errors=20;
    
    /* Enable ODS TRACE to see available tables */
    ods trace on;
    
    /* Capture LSMeans output */
    ods output LSMeans=work.lsmeans;
    
    /* Run PROC GLM with LSMeans */
    proc glm data=work.testdata;
        class Treatment;
        model TumorSize = Treatment;
        lsmeans Treatment;
    run;
    
    /* Verify LSMeans dataset was created */
    proc contents data=work.lsmeans; 
    run;
    
    /* Print LSMeans dataset for verification */
    proc print data=work.lsmeans; 
    run;
    
    /* Show ODS trace information */
    ods trace off;
    """
    
    result = sas.submit(sas_code)
    logger.info("LSMeans analysis completed")
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
    
    # Step 7: Verify LSMeans dataset exists in Python
    logger.info("Step 7: Verifying LSMeans dataset can be extracted to Python")
    try:
        if sas.exist('lsmeans'):
            lsmeans_df = sas.sd2df('lsmeans')
            logger.info(f"LSMeans dataset extracted: {lsmeans_df.shape}")
            logger.info(f"LSMeans columns: {list(lsmeans_df.columns)}")
            logger.info("LSMeans data:")
            logger.info(lsmeans_df.to_string())
            
            # Additional analysis of LSMeans data
            if not lsmeans_df.empty:
                logger.info("LSMeans summary:")
                logger.info(f"  - Number of treatment groups: {len(lsmeans_df)}")
                if 'Treatment' in lsmeans_df.columns:
                    logger.info(f"  - Treatment groups: {list(lsmeans_df['Treatment'].unique())}")
                if 'LSMean' in lsmeans_df.columns:
                    logger.info(f"  - LSMeans range: {lsmeans_df['LSMean'].min():.4f} to {lsmeans_df['LSMean'].max():.4f}")
        else:
            logger.error("LSMeans dataset not found in SAS")
    except Exception as e:
        logger.error(f"Failed to extract LSMeans dataset: {e}")
    
    # Step 8: Test additional LSMeans options
    logger.info("Step 8: Testing additional LSMeans options")
    extended_lsmeans_code = """
    /* Test LSMeans with additional options */
    ods output LSMeanDiffCL=work.lsmeans_diff;
    
    proc glm data=work.testdata;
        class Treatment;
        model TumorSize = Treatment;
        lsmeans Treatment / stderr pdiff cl adjust=bon;
    run;
    
    /* Check if difference dataset was created - with explicit output capture */
    ods output Contents=work.contents_lsmeans_diff;
    proc contents data=work.lsmeans_diff; 
    run;
    ods output close;
    
    /* Print dataset with explicit output capture */
    ods output Print=work.print_lsmeans_diff;
    proc print data=work.lsmeans_diff; 
    run;
    ods output close;
    
    /* Also try direct print to log */
    proc print data=work.lsmeans_diff;
        title "LSMeans Differences Dataset - Direct Print";
    run;
    """
    
    extended_result = sas.submit(extended_lsmeans_code)
    logger.info("Extended LSMeans analysis completed")
    logger.info("Extended analysis log:")
    logger.info(extended_result['LOG'])
    
    # Step 8.5: Verify lsmeans_diff dataset exists and extract it
    logger.info("Step 8.5: Verifying lsmeans_diff dataset")
    try:
        if sas.exist('lsmeans_diff'):
            logger.info("✅ lsmeans_diff dataset exists in SAS")
            
            # Extract the dataset to Python
            lsmeans_diff_df = sas.sd2df('lsmeans_diff')
            logger.info(f"lsmeans_diff dataset extracted: {lsmeans_diff_df.shape}")
            logger.info(f"lsmeans_diff columns: {list(lsmeans_diff_df.columns)}")
            logger.info("lsmeans_diff data:")
            logger.info(lsmeans_diff_df.to_string())
            
            # Additional analysis
            if not lsmeans_diff_df.empty:
                logger.info("lsmeans_diff summary:")
                logger.info(f"  - Number of pairwise comparisons: {len(lsmeans_diff_df)}")
                if 'Treatment' in lsmeans_diff_df.columns and '_Treatment_' in lsmeans_diff_df.columns:
                    logger.info(f"  - Treatment comparisons: {lsmeans_diff_df['Treatment'].unique()} vs {lsmeans_diff_df['_Treatment_'].unique()}")
                if 'Estimate' in lsmeans_diff_df.columns:
                    logger.info(f"  - Difference estimates range: {lsmeans_diff_df['Estimate'].min():.4f} to {lsmeans_diff_df['Estimate'].max():.4f}")
        else:
            logger.error("❌ lsmeans_diff dataset not found in SAS")
            
            # Check what datasets are actually available
            logger.info("Checking what datasets are available in SAS:")
            try:
                all_datasets = sas.sasdata2dataframe('_all_')
                logger.info(f"Available datasets: {all_datasets}")
            except Exception as e:
                logger.warning(f"Could not get list of all datasets: {e}")
                
    except Exception as e:
        logger.error(f"Failed to extract lsmeans_diff dataset: {e}")
    
    # Step 8.6: Try to get contents and print output as separate datasets
    logger.info("Step 8.6: Extracting contents and print output as datasets")
    try:
        if sas.exist('contents_lsmeans_diff'):
            contents_df = sas.sd2df('contents_lsmeans_diff')
            logger.info("Contents of lsmeans_diff dataset:")
            logger.info(contents_df.to_string())
        else:
            logger.warning("contents_lsmeans_diff dataset not found")
            
        if sas.exist('print_lsmeans_diff'):
            print_df = sas.sd2df('print_lsmeans_diff')
            logger.info("Print output of lsmeans_diff dataset:")
            logger.info(print_df.to_string())
        else:
            logger.warning("print_lsmeans_diff dataset not found")
    except Exception as e:
        logger.error(f"Failed to extract contents/print datasets: {e}")
    
    # Step 8.7: Alternative approach - use proc export to see the data
    logger.info("Step 8.7: Using proc export to verify lsmeans_diff data")
    export_code = """
    /* Export lsmeans_diff to a text file to see the data */
    proc export data=work.lsmeans_diff
        outfile='/tmp/lsmeans_diff_export.txt'
        dbms=tab;
    run;
    
    /* Also try proc univariate to see summary statistics */
    proc univariate data=work.lsmeans_diff;
        var Estimate;
        title "Summary Statistics for LSMeans Differences";
    run;
    """
    
    export_result = sas.submit(export_code)
    logger.info("Export test completed")
    logger.info("Export test log:")
    logger.info(export_result['LOG'])
    
    # Check if export file was created
    if os.path.exists('/tmp/lsmeans_diff_export.txt'):
        logger.info("✅ lsmeans_diff export file created")
        try:
            with open('/tmp/lsmeans_diff_export.txt', 'r') as f:
                export_content = f.read()
            logger.info("Exported lsmeans_diff data:")
            logger.info(export_content)
        except Exception as e:
            logger.error(f"Failed to read export file: {e}")
    else:
        logger.warning("❌ lsmeans_diff export file not found")
    
    # Step 9: Test ODS output for LSMeans
    logger.info("Step 9: Testing ODS PDF/RTF output for LSMeans")
    ods_lsmeans_code = """
    /* Test ODS output with LSMeans */
    ods pdf file='/tmp/lsmeans_test.pdf' style=journal;
    ods rtf file='/tmp/lsmeans_test.rtf' style=journal;
    
    proc glm data=work.testdata;
        class Treatment;
        model TumorSize = Treatment;
        lsmeans Treatment / stderr pdiff cl adjust=bon;
    run;
    
    proc print data=work.lsmeans;
        title "LSMeans Results";
    run;
    
    ods pdf close;
    ods rtf close;
    """
    
    ods_result = sas.submit(ods_lsmeans_code)
    logger.info("ODS LSMeans test completed")
    logger.info("ODS test log:")
    logger.info(ods_result['LOG'])
    
    # Check if test files were created
    if os.path.exists('/tmp/lsmeans_test.pdf'):
        logger.info("✅ LSMeans PDF file created successfully")
    else:
        logger.warning("❌ LSMeans PDF file not found")
    
    if os.path.exists('/tmp/lsmeans_test.rtf'):
        logger.info("✅ LSMeans RTF file created successfully")
    else:
        logger.warning("❌ LSMeans RTF file not found")
    
    logger.info("=" * 60)
    logger.info("LSMEANS TEST - COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
except Exception as e:
    logger.error(f"LSMeans test failed with exception: {e}")
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
    print(f"\nLSMeans test completed. Check log file: {log_file}")