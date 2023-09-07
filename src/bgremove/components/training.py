#############################################################################################
# Imports 
###############################################################################################
import tensorflow as tf
import cv2
from glob import glob
import numpy as np
import os
import datetime
from bgremove.constants import *
from bgremove.utils.common import read_yaml
from bgremove.entity.config_entity import TrainingConfig


############################################################################################
# Global Varieables
# ##########################################################################################
params = read_yaml(PARAMS_FILE_PATH)
config_path = read_yaml(CONFIG_FILE_PATH)
H = params.HEIGHT
W = params.WIDTH


# Get the current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

save_model_file_path =f'{config_path.training.trained_model_path}_{timestamp}.h5'
###################################################################################################
# Class Training 
######################################################################################################
class Training:
    def __init__(self, config:TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.base_model_path
        )
        return self.model

    def load_data(self, path):
        train_x = sorted(glob(os.path.join(path, "train", "blurred_image", "*.jpg")))
        train_y = sorted(glob(os.path.join(path, "train", "mask", "*.png")))

        valid_x = sorted(glob(os.path.join(path, "validation", "P3M-500-NP", "original_image", "*.jpg")))
        valid_y = sorted(glob(os.path.join(path, "validation", "P3M-500-NP", "mask", "*.png")))
        
        print(f"traiing list {len(train_x)}, and trainy {len(train_y)}, validation is {len(valid_x)}& {len(valid_y)}")

        return (train_x, train_y), (valid_x, valid_y)
    

    def read_image(self, path):
        path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (W, H))
        x = x / 255.0
        x = x.astype(np.float32)
        return x

    def read_mask(self, path):
        path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (W, H))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=-1)
        return x

    def tf_parse(self, x, y):
        def _parse(x, y):
            x = self.read_image(x)
            y = self.read_mask(y)
            return x, y

        x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
        x.set_shape([H, W, 3])
        y.set_shape([H, W, 1])
        return x, y

    def tf_dataset(self, X, Y, batch):
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        ds = ds.map(self.tf_parse).batch(batch).prefetch(10)
        print(f"the whole df {ds}")
        return ds

    def loading_data_set(self, path, batch_size):
        (train_x, train_y), (valid_x, valid_y) = self.load_data(path)

        print(f"Train: {len(train_x)} - {len(train_y)}")
        print(f"Valid: {len(valid_x)} - {len(valid_y)}")
        
        print(batch_size)

        train_dataset = self.tf_dataset(train_x, train_y, batch=batch_size)
        valid_dataset = self.tf_dataset(valid_x, valid_y, batch=batch_size)
        return (train_dataset, valid_dataset)

        
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    
    def train(self, trained_ds, valid_data, callback_list):
        self.model.fit(
            trained_ds,
            initial_epoch= params.INITIAL_EPOCHS,
            epochs=params.EPOCHS,
            validation_data= valid_data,
            callbacks= callback_list,
            verbose=params.VERBOSE
        )
        
        self.save_model(
            path = save_model_file_path,
            model= self.model
            )

        