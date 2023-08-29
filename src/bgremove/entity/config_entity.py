from dataclasses import dataclass
from pathlib import Path

###########################################################
#
# Data Ingestion / Data Set Download Files 
#
###########################################################
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path


#############################################################
# 
#  Modeling ENTITY CLASS
# 
# ############################################################


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir:Path
    base_model_path:Path
    updated_base_model_path:Path
    params_image_size:list
    params_learning_rate:float
    params_weights:str
    params_classes:int