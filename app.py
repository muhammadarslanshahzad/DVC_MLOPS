from flask import Flask, request, jsonify, send_file, render_template
import os
import cv2
import numpy as np
import tensorflow as tf
from bgremove.pipeline.predict import PredictPipeline
from PIL import Image
from datetime import datetime
import concurrent.futures

###################################################
# Global Variables
# ###############################################
app = Flask(__name__)
model_path = "assets/model/model.h5"
output_path = "assets/results/"
host = "172.16.2.231"
port = 9000

###########################################################
# loading model & calling prediction pipeline
# ######################################################
prediction_pipeline = PredictPipeline(model_path)

########################################################
#  processing & Prediction
######################################################### #
def process_image(image_PIL, image_name):
    image = np.array(image_PIL)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    np.random.seed(42)
    tf.random.set_seed(42)
    print(f"========================> Processing Image =========>")
    prediction_pipeline.predict(image, image_name)

###############################################################
# Routes & API
# ##############################################################

@app.route('/', methods=["GET", "POST"])
def home():
    return render_template('index.html')

@app.route('/predict', methods=["GET","POST"])
def prediction():

    req_image = request.files['image']
    image_PIL = Image.open(req_image)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_name = req_image.filename.replace(".jpg","")+'_'+current_time+".png"
   
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(process_image, image_PIL, image_name)

    return_image_path = f"http://{host}:{port}/{output_path+image_name}"
    response = {"processed_image_url":return_image_path}
    print(response)
    return jsonify(response)


@app.route("/assets/results/<filename>")
def serve_image(filename):
    # Replace 'path/to/' with the correct path to your image folder.
    image_path = os.path.join("assets/results/", filename)
    print(image_path)

    if not os.path.isfile(image_path):
        # Return an error image or a default image if the requested image doesn't exist.
        default_image_path = "assets/error.jpeg"  # Replace with the path to your default image.
        image_path = default_image_path

    # Determine the image mimetype based on the file extension.
    # You may need to handle additional image formats as needed.
    mimetype = "image/jpeg" if filename.lower().endswith(".jpg") else "image/png"
    print(f'Returning the image {image_path}')
    return send_file(image_path, mimetype=mimetype)

if __name__ == '__main__':
    app.run(host=host, port=port, debug=True)
    