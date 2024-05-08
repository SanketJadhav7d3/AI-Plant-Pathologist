

import tensorflow as tf
import requests
import os
import base64
from PIL import Image
import numpy as np
from io import BytesIO
from datetime import timezone, datetime
from google.cloud import datastore
from google.cloud import storage
import json

classes_to_index = {'Apple___Apple_scab': 0,
                  'Apple___Black_rot': 1,
                  'Apple___Cedar_apple_rust': 2,
                  'Apple___healthy': 3,
                  'Blueberry___healthy': 4,
                  'Cherry_(including_sour)___Powdery_mildew': 5,
                  'Cherry_(including_sour)___healthy': 6,
                  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,
                  'Corn_(maize)___Common_rust_': 8,
                  'Corn_(maize)___Northern_Leaf_Blight': 9,
                  'Corn_(maize)___healthy': 10,
                  'Grape___Black_rot': 11,
                  'Grape___Esca_(Black_Measles)': 12,
                  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13,
                  'Grape___healthy': 14,
                  'Orange___Haunglongbing_(Citrus_greening)': 15,
                  'Peach___Bacterial_spot': 16,
                  'Peach___healthy': 17,
                  'Pepper,_bell___Bacterial_spot': 18,
                  'Pepper,_bell___healthy': 19,
                  'Potato___Early_blight': 20,
                  'Potato___Late_blight': 21,
                  'Potato___healthy': 22,
                  'Raspberry___healthy': 23,
                  'Soybean___healthy': 24,
                  'Squash___Powdery_mildew': 25,
                  'Strawberry___Leaf_scorch': 26,
                  'Strawberry___healthy': 27,
                  'Tomato___Bacterial_spot': 28,
                  'Tomato___Early_blight': 29,
                  'Tomato___Late_blight': 30,
                  'Tomato___Leaf_Mold': 31,
                  'Tomato___Septoria_leaf_spot': 32,
                  'Tomato___Spider_mites Two-spotted_spider_mite': 33,
                  'Tomato___Target_Spot': 34,
                  'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35,
                  'Tomato___Tomato_mosaic_virus': 36,
                  'Tomato___healthy': 37}

indices_to_class = {v: k for k, v in classes_to_index.items()}

cwd = os.path.abspath(os.path.dirname(__file__))

# model_path = os.path.abspath(os.path.join(cwd, "model_checkpoint.h5"))

# model = tf.keras.models.load_model(model_path)



if __name__ == "__main__":
    path = "static/css/images/demo-image-2.jpeg"
    
    signed_url = upload_image("uploaded-images-inference", path, "something.jpeg") 
