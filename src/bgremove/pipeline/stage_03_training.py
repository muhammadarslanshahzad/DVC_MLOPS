##################################################################
#   IMPORTS
##################################################################

from bgremove import logger
from bgremove.config.configuration import ConfigurationManager
from bgremove.components.base_model import PrepareBaseModel
from bgremove.utils.common import read_yaml
from bgremove.constants import *
from bgremove.components.prepare_callbacks import PrepareCallBacks
from bgremove.components.training import Training


########################################################################
# Global Vairables
# #######################################################################

STAGE_NAME = "Training MODEL STAGE"


config = read_yaml(CONFIG_FILE_PATH)
params = read_yaml(PARAMS_FILE_PATH)
save_model_path = config.prepare_base_model.base_model_path
data_set_path = config.data_ingestion.unzip_data_dir

batach_size = params.BATCH_SIZE

###########################################################################
# Training Pipeline
# ########################################################################

class TrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_call_backs_config()
        prepare_callbacks = PrepareCallBacks(config= prepare_callbacks_config)
        callback_list = prepare_callbacks.get_callbacks()
        
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        train_ds, valid_ds = training.loading_data_set(data_set_path, batach_size)
        print(train_ds)
        training.train(train_ds, valid_ds, callback_list)
        


if __name__ == '__main__':
    try:
        logger.info(f'=====================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<================================================')
        obj = TrainingPipeline()
        obj.main()
        logger.info(f'=============================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<<<<<<<<<==============================================\n\n\n x=============================================================================================================================================x')
        
    except Exception as e:
        logger.exception(e)
        raise e