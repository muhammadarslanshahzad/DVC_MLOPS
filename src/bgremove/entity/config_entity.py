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



##############################################################
# 
#   Call Backs
# 
# ##########################################################################


@dataclass(frozen=True)
class PrepareCallBackConfig:
    root_dir:Path
    tensorboard_root_log_dir:Path
    checkpoint_model_filepath:Path
    csv_filePath:Path


#####################################################################################
# 
# Training Class Config
# 
# ##################################################################################

@dataclass(frozen=True)
class TrainingConfig:
    root_dir:Path
    trained_model_path:Path
    base_model_path:Path
    training_data:Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
        