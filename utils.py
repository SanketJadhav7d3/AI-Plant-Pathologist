

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


storage_client = storage.Client.from_service_account_json("key.json")

datastore_client = datastore.Client.from_service_account_json("datastore.json", project="plantdiseaseclassifierfrontend")


def upload_image(bucket_name, image, destination_blob_name):

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_file(image)

    print(f"Image uploaded to gs://{bucket_name}/{destination_blob_name}")

    expiration = int(datetime.now(tz=timezone.utc).timestamp()) + 3600

    signed_url = blob.generate_signed_url(expiration=expiration)

    return signed_url

def save_model_inference(signed_url, labels):

    data = { 'signed_url' : signed_url, 'classes' : labels }

    json_data = json.dumps(data)

    entity_key = datastore_client.key("Results", "result")

    entity = datastore.Entity(key=entity_key)

    for key, value in data.items():
        if isinstance(value, list) and all(isinstance(item, list) for item in value):
            entity[key] = json.dumps(value)
        else:
            entity[key] = value

    datastore_client.put(entity)

    print("sent data to firestore")

def retrieve_results():
    entity_key = datastore_client.key("Results", "result")
    entity = datastore_client.get(entity_key)

    return entity

def download_image(imageurl):
    '''
    returns numpy array of image
    '''
    response = requests.get(imageurl)
    image = Image.open(BytesIO(response.content))
    return np.array(image).astype(np.float32)

def preprocess_image(image_array):

    image_array = tf.image.resize(image_array, (200, 200)).numpy()

    image_array = tf.keras.applications.resnet50.preprocess_input(image_array)

    return image_array


if __name__ == "__main__":
    path = "static/css/images/demo-image-2.jpeg"
    
    signed_url = upload_image("uploaded-images-inference", path, "something.jpeg") 
