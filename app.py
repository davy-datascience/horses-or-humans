import io
import json
import os
import urllib
from urllib.request import urlopen, Request
from io import BytesIO
from PIL import Image
import requests

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from classes.prediction import Prediction, get_predictions_json
import logging
import uuid

app = Flask(__name__)
app.secret_key = b'c\x83\xc5MU\x9eV\xd2R\x12\x87(\x8c\xb6S\xf3H\xbc\xedn\xa3\xa3\tl'
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.debug = True


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict')
def predict():
    predictions = json.loads(session['predictions'])
    return render_template('predict.html', predictions=predictions, title="Predictions")


def upload_file(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = Request(url=url, headers=headers)
    fd = urllib.request.urlopen(req)
    image_file = io.BytesIO(fd.read())
    im = Image.open(image_file)
    im = im.resize((300, 300), Image.ANTIALIAS)

    return im


@app.route('/', methods=['POST'])
def upload():
    bucket = "horses-or-humans"
    model = load_model('model.h5')
    predictions = []
    for f in request.files.getlist('file'):
        # Save image to s3
        s3 = boto3.client('s3')
        s3.put_object(Bucket="horses-or-humans", Key=f.filename, Body=f, ACL='public-read', ContentType='image/jpeg')

        img = upload_file("https://horses-or-humans.s3.eu-west-3.amazonaws.com/" + f.filename)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        prediction = None
        if classes[0] > 0.5:
            print(f.filename + " is a human")
            prediction = Prediction(f.filename, "human")
        else:
            print(f.filename + " is a horse")
            prediction = Prediction(f.filename, "horse")
        predictions.append(prediction)
    predictions_json = get_predictions_json(predictions)
    predictions_str = json.dumps(predictions_json)
    session['predictions'] = predictions_str
    return redirect(url_for('predict'))
