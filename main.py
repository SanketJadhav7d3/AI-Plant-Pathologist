
from flask import Flask, render_template, request, flash, jsonify, url_for, redirect, make_response, session
from markupsafe import Markup
from utils import indices_to_class
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
from functools import wraps
import cv2 
 

app = Flask(__name__)

app.config['SECRET_KEY'] = "hello_everynyan"
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image'

trial_images_folder_path = './static/css/images'

dropzone = Dropzone(app)

upload_folder_path = os.path.join("static", "uploads")

app.config['UPLOAD_FOLDER'] = upload_folder_path

try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
except:
    print("key not found")

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-pro')

chat_api = model.start_chat(history=[])

# load the model
classifier = tf.saved_model.load('./model') 

def form_submission_required(func):
    wraps(func)

    def secure_function(*args, **kwargs):
        if 'form_submitted' not in session or not session['form_submitted']:
            return redirect(url_for("trial"))
        return func(*args, **kwargs)

    return secure_function


def get_response_from_gemni(most_likely_diseases):
    prompt = f"You are virtual plant pathologist and will give responses related to it only. The output of a classifcation model along with logits is {most_likely_diseases}\
            Explain the above inference neatly giving the information about the diseases\
            and briefly explain about the most likely disease and causes of it\
            Here give more focus on the most likely disease and don't explain that there is tuple\
            Give response which is understandable by non-machine learning people"

    response = model.generate_content(prompt)

    response_str = ""

    try:
        for chunk in response:
            response_str += chunk.text
    except:
        print("something went wrong")
    
    return mistune.html(response_str)


def func_chat(prompt):
    response = chat_api.send_message(prompt)

    response_str = ""

    try:
        for chunk in response:
            response_str += chunk.text
    except:
        print("something went wrong")

    return response_str


@app.route('/chat', methods=["POST"])
def chat():

    if request.method == "POST":
        prompt = request.form["prompt"]

        response = chat_api.send_message(prompt)

        response_str = ""

        try:
            for chunk in response:
                response_str += chunk.text
        except:
            print("something went wrong")

        response_html_str = mistune.html(response_str)

        json_response = json.dumps({
            "response" : response_html_str
            })

        return json_response

    return "Something went wrong"


def remove_underscore(name):
    return re.sub(r'_+', ' ', name)

def classify(img):
    x_test = np.asarray(img).astype(np.float32)

    x_test = tf.image.resize(x_test, (200, 200)).numpy()

    x_test = tf.keras.applications.resnet50.preprocess_input(x_test)

    x_test = tf.expand_dims(x_test, 0)

    predictions = classifier(x_test)

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

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(file_path)
        except Exception as e:
            print(e)
            redirect(url_for('trial'))

        return redirect(url_for('results', filename=filename))

    redirect(url_for('trial'))

@app.route('/diseases', methods=['GET'])
def diseases():
    return render_template("diseases.html")

@form_submission_required
@app.route('/results', methods=['GET'], endpoint='results')
def results():

    filename = request.args.get('filename')

    file_path = os.path.join('static', f'uploads/image.jpeg')

    img = Image.open(file_path)

    labels_ = classify(img)

    prompt = f"You are virtual plant pathologist and will give responses related to it only. The output of a classifcation model along with logits is {labels_}\
            Explain the above inference neatly giving the information about the diseases\
            and briefly explain about the most likely disease and causes of it\
            Here give more focus on the most likely disease and don't explain that there is tuple\
            Give response which is understandable by non-machine learning people"

    response = func_chat(prompt)

    response_html = mistune.html(response)

    response_html = Markup(response_html)

    return render_template("results.html", img_path=file_path, response=response, response_html=response_html)


@app.route('/trial', methods=['GET'])
def trial():
    return render_template("trialpage.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
