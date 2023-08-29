###########################################################
# IMports
############################################################

from bgremove import logger
from bgremove.pipeline.stage_01_data_ingestion import DataIngestionPipeline


##################################################
# Data Ingestion Stage Pipeline
#####################################################
STAGE_NAME = 'Data Ingestion stage'
try:
    logger.info(f'===================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<<<================================================')
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f'===================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<<<<<<<===========================================\n\n\n xxxxxxxxxxxx===================================================================================================================xxxxxxxxxxxxxx')
except Exception as e:
    logger.exception(e)
    raise e

