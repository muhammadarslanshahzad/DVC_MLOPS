##################################################################
#   IMPORTS
##################################################################

from bgremove import logger
from bgremove.config.configuration import ConfigurationManager
from bgremove.components.base_model import PrepareBaseModel
from bgremove.utils.common import read_yaml
from bgremove.constants import CONFIG_FILE_PATH



STAGE_NAME = "PREPARING MODEL STAGE"

config_model = read_yaml(CONFIG_FILE_PATH)
save_model_path = config_model.prepare_base_model.base_model_path



class BaseModelPrepPipeline:
    def __init__(self) -> None:
        pass  
        

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        model = prepare_base_model.get_base_model()
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

