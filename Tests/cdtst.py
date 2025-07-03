import saspy
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting captive data SAS test")
sas = saspy.SASsession()
try:
    # Step 1: Create captive data
    logger.info("Step 1: Creating captive data")
    captive_data = pd.DataFrame({
        'Treatment': ['High Dose', 'High Dose', 'Low Dose', 'Low Dose', 'Placebo', 'Placebo'],
        'TumorSize': [1300, 1302, 5800, 5840, 9200, 9280]
    })
    logger.info(f"Captive data shape: {captive_data.shape}")
    logger.info(f"Columns: {captive_data.columns.tolist()}")
    logger.info(f"First few rows:\n{captive_data}")

    # Step 2: Transfer data to SAS
    logger.info("Step 2: Transferring data to SAS")
    sas.df2sd(captive_data, 'testdata')
    logger.info("Data transferred to SAS table 'work.testdata'")

    # Step 3: Run SAS analysis
    logger.info("Step 3: Running SAS analysis")
    result = sas.submit("""
    options errors=20;
    ods trace on;
    ods output 
        LSMeans=work.lsmeans 
        LSMeanDiffCL=work.diffs;
    proc glm data=work.testdata;
        class Treatment;
        model TumorSize = Treatment;
        lsmeans Treatment / stderr pdiff cl adjust=bon;
    run;
    proc contents data=work.lsmeans; run;
    proc contents data=work.diffs; run;
    proc print data=work.lsmeans; run;
    proc print data=work.diffs; run;
    ods trace off;
    """)
    logger.info("Step 4: SAS execution log")
    logger.info(result['LOG'])
except Exception as e:
    logger.error(f"Error: {e}")
finally:
    logger.info("Cleaning up SAS session")
    sas.endsas()
    logger.info("SAS session closed")