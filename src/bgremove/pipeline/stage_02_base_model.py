##################################################################
#   IMPORTS
##################################################################

from bgremove import logger
from bgremove.config.configuration import ConfigurationManager
from bgremove.components.base_model import PrepareBaseModel
from bgremove.utils.common import read_yaml
from bgremove.constants import *



STAGE_NAME = "PREPARING MODEL STAGE"

params = read_yaml(PARAMS_FILE_PATH)
config_model = read_yaml(CONFIG_FILE_PATH)
save_model_path = config_model.prepare_base_model.base_model_path



class BaseModelPrepPipeline:
    def __init__(self) -> None:
        pass  
        

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        model = prepare_base_model.u2net(params.IMAGE_SIZE, params.OUT_CH, params.INT_CH, num_classes=1)
        model.summary()
        model.save(save_model_path)


if __name__ == '__main__':
    
    try:
        logger.info(f'=====================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<================================================')
        obj = BaseModelPrepPipeline()
        obj.main()
        logger.info(f'=============================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<<<<<<<<<<<<<<<==============================================\n\n\n x=============================================================================================================================================x')
        
    except Exception as e:
        logger.exception(e)
        raise e

