##################################################################
#   IMPORTS
##################################################################

from bgremove.config.configuration import ConfigurationManager
from bgremove.components.data_ingestion import DataIngestion
from bgremove import logger

STAGE_NAME = "Data Ingestion Stage"

#####################################################################
# Pipeline Class 
#####################################################################

class DataIngestionPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config= data_ingestion_config)
        # data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        # print("\n\n\n\n\n the data file will download here ====================>>>>>>>>>>>>\n\n\n\n\n\n")



if __name__ == '__main__':
    try:
        logger.info(f'=====================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<================================================')
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f'=============================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<<<<<<<<<==============================================\n\n\n x=============================================================================================================================================x')
        
    except Exception as e:
        logger.exception(e)
        raise e