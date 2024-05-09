# Plant Leaf Disease Classifier using ResNet50

This repository contains a deep learning classifier trained on the [Plant Leaf Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) consisting of 38 classes of plant leaf diseases. 

## Model 

A pre-trained Resnet-50 model was fine-tuned on the Plant Leaf Disease Dataset.

## Deployment

* The trained classifier has been deployed to the Google Vertex AI Endpoint. 
* The flask frontend is hosted on google cloud run.
* Gogole cloud storage is also used to store the uploaded images.

Click here to try

https://classifier-guzxsmcyia-el.a.run.app (currently doesn't work)

# Tensorflow savedmodel format drive link

https://drive.google.com/drive/folders/1--JgkNaDHVVI2RY5CscyuZ7C-upJxsu0?usp=sharing

# Steps to locally

* First obtain gemini api key and set it as your environment variable.
* Download the tensorflow savedmodel format and store the contents in the model directory.
* Create python virtual environment with the tool of your choice.
* Download the requirements

```bash
pip install -r requirements.txt
```
* Run the app
  
```bash
python main.py
```

