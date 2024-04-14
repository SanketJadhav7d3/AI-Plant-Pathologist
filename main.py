
from flask import Flask, render_template, request, flash, jsonify, url_for, redirect, make_response
from markupsafe import Markup
from utils import upload_image, save_model_inference, retrieve_results, indices_to_class
import json
import random
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
import requests
import os
import re
import sys
import numpy as np
from google.cloud import aiplatform
from io import BytesIO
import tensorflow as tf
from PIL import Image
import google.generativeai as genai
import mistune


app = Flask(__name__)

app.config['SECRET_KEY'] = "hello_everynyan"
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image'

trial_images_folder_path = './static/css/images'

dropzone = Dropzone(app)

upload_folder_path = os.path.join("static", "uploads")

app.config['UPLOADED_PATH'] = upload_folder_path

# endpoint 
ENDPOINT_ID = 8950473250840772608
PROJECT_NUMBER = 417525863877

endpoint_name = f"projects/{PROJECT_NUMBER}/locations/asia-southeast1/endpoints/{ENDPOINT_ID}"

endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)

try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
except:
    print("key not found")

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-pro')


def get_response_from_gemni(most_likely_diseases):
    prompt = f"The output of a classifcation model along with logits is {most_likely_diseases}\
            Explain the above inference neatly giving the information about the diseases\
            and briefly explain about the most likely disease and causes of it"


    response = model.generate_content(prompt)

    response_str = ""

    try:
        for chunk in response:
            response_str += chunk.text
    except:
        print("something went wrong")
    

    return mistune.html(response_str)


def remove_underscore(name):
    return re.sub(r'_+', ' ', name)

def classify(instances : list):
    file = instances[0]

    img = Image.open(BytesIO(file.read()))

    img_array = np.array(img)

    x_test = np.asarray(img_array).astype(np.float32)

    x_test = tf.image.resize(x_test, (200, 200)).numpy()

    x_test = tf.keras.applications.resnet50.preprocess_input(x_test)


    x_test = x_test.tolist()

    predictions = endpoint.predict(instances=[x_test]).predictions

    infer_class = np.argsort(predictions[0])[::-1][:5]

    return [[remove_underscore(indices_to_class[i]), predictions[0][i]] for i in infer_class]


@app.route('/', methods=['GET', 'POST'])
def home():
    # else just render the basic template 
    return render_template("homepage.html")

@app.route('/infer', methods=['GET', 'POST'])
def infer():
    if request.method == 'POST':
        file = request.files.get('file')

        if file:
            filename = secure_filename(file.filename)
            
            try:
                signed_url = upload_image("uploaded-images-inference", file, filename) 
            except Exception as e:
                print(e)
                redirect(url_for('trial'))

            file.seek(0)

            labels = classify([file])


            try:
                save_model_inference(signed_url, labels)
                pass 
            except Exception as e:
                print(e)
                redirect(url_for('trial'))

    return redirect(url_for('results'))

@app.route('/results', methods=['GET'])
def results():

    data = retrieve_results()

    labels = json.loads(data['classes'])
    signed_url = data['signed_url']

    response = Markup(get_response_from_gemni(labels))


    return render_template("results.html", img_path=signed_url, labels=labels, response=response)

@app.route('/trial', methods=['GET'])
def trial():
    return render_template("trialpage.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
