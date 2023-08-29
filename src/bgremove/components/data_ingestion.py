###################################
#  IMports
###################################
import os
import urllib.request as request
import zipfile
import gdown
from bgremove import logger
from bgremove.utils.common import get_size
from bgremove.entity.config_entity import DataIngestionConfig

################################
# Data Ingestion Class 
######################


class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        '''
            Dowloading the Data From the Drive
        '''
        if not os.path.exists(self.config.local_data_file):
            gdown.download(self.config.source_URL, self.config.local_data_file, quiet=False)
            file_name= self.config.local_data_file    
            logger.info(f"{file_name}Data has downlaod! with following info:\n")
        else:
            logger.info(f"File already exist of size: {get_size(Path(self.config.local_data_file))}")
            
    def extract_zip_file(self):
        '''
            zip file extraction
            zip file path:str
            Extracts the zip file into the data directory
            Function return None
        '''
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
                