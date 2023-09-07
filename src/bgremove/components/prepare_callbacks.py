######################################################
#
# IMPORTS 
#
#####################################################

import os
import tensorflow as tf
import time
from bgremove.constants import *
from bgremove.utils.common import read_yaml
from bgremove.config.configuration import PrepareCallBackConfig
import datetime

params = read_yaml(PARAMS_FILE_PATH)
config_path = read_yaml(CONFIG_FILE_PATH)

# Get the current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

chk_point_file_path =f'{config_path.prepare_callbacks.checkpoint_model_filepath}_{timestamp}.h5'
##########################################################
# Call Back Class
#########################################################

class PrepareCallBacks:
    def __init__(self, config:PrepareCallBackConfig):
        self.config= config
    
    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}"
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    
    @property
    def _create_ckpt_callbacks(self):
            return tf.keras.callbacks.ModelCheckpoint(
                filepath=chk_point_file_path,
                verbose=params.VERBOSE,
                save_best_only=params.SAVE_BEST_ONLY)
    
    @property
    def _reduce_on_paleatue(self):
            return tf.keras.callbacks.ReduceLROnPlateau(
                monitor=params.MONITOR,
                factor=params.FACTOR, 
                patience=params.PATIENCE_REDUCE_LEARNING, 
                min_lr=params.MIN_LR, 
                verbose=params.VERBOSE)
            
    @property
    def _csv_logger(self):
            return tf.keras.callbacks.CSVLogger(self.config.csv_filePath)
    
    @property 
    def _early_stopping(self):
            return tf.keras.callbacks.EarlyStopping(monitor=params.MONITOR, patience=params.PATIENCE_EARLY_STOPPING, restore_best_weights=params.RESTORE_BEST_WEIGHTS)
            
    def get_callbacks(self):
        return [
            self._create_tb_callbacks, 
            self._create_ckpt_callbacks,
            self._reduce_on_paleatue,
            self._csv_logger,
            self._early_stopping
        ]