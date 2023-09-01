import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm

########################################################
# Prediction Pipeline
####################################################### ##
H=256
W= 256
class PredictPipeline:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        # self.H = 256
        # self.W = 256

   
    def predict(self, image_path,image_name, output_dir="/home/ubuntu/ArslanShahzad/bgremoval/assets/results"):
        """ Reading the image """
        image = image_path
        x = cv2.resize(image, (H, W))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        pred = self.model.predict(x, verbose=0)


        """ Joint and save mask """
        pred_list = []
        for item in pred:
            p = item[0] * 255
            p = np.concatenate([p, p, p], axis=-1)

            pred_list.append(p)

        name = os.path.basename(image_name)
        mask_image_path = os.path.join(output_dir, "mask", name)
        cat_images = np.concatenate(pred_list, axis=1)
        cv2.imwrite(mask_image_path, cat_images)

        """ Save final mask """
        image_h, image_w, _ = image.shape

        y0 = pred[0][0]
        y0 = cv2.resize(y0, (image_w, image_h))
        y0 = np.expand_dims(y0, axis=-1)
        
        # y0 = np.concatenate([y0, y0, y0], axis=-1)
        # for making image transparent
        alpha_mask = (y0*255).astype(np.uint8)
        
        #crete a 4 channel RGBA image
        image_final = np.zeros((image_h, image_w, 4), dtype=np.uint8)
        image_final[:, :, :3] = image
        image_final[:, :, 3] = alpha_mask.squeeze()
         


        # cat_images = np.concatenate([image, y0 * 255, image * y0], axis=1)
        joint_image_path = os.path.join(output_dir, name[:-4]+".png")
        cv2.imwrite(joint_image_path, image_final)
        
        
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    predictor = PredictPipeline("/home/ubuntu/ArslanShahzad/bgremoval/artifacts/prepare_callbacks/checkpoint_dir/model.h5")

    test_images = glob("/home/ubuntu/ArslanShahzad/bgremoval/test/*.jpg")
    print(f" the total images in the folders are ======> {len(test_images)}")
    for image_path in tqdm(test_images, total=len(test_images)):
        print(f"performing the perdiction of the {image_path}")
        image_name = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        image = np.array(image_name)
        predictor.predict(image, image_path)
