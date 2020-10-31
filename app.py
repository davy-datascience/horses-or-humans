import json
import os
import boto3
from botocore.config import Config
from flask import Flask, render_template, request, redirect, url_for, session

import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from classes.prediction import Prediction, get_predictions_json

app = Flask(__name__)
app.secret_key = b'c\x83\xc5MU\x9eV\xd2R\x12\x87(\x8c\xb6S\xf3H\xbc\xedn\xa3\xa3\tl'
app.config['UPLOAD_FOLDER'] = 'static/uploads'


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict')
def predict():
    predictions = json.loads(session['predictions'])
    return render_template('predict.html', predictions=predictions, title="Predictions")


@app.route('/', methods=['POST'])
def upload_file():
    model = load_model('model.h5')
    predictions = []
    for uploaded_file in request.files.getlist('file'):
        if uploaded_file.filename != '':
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename))
            img = image.load_img("static/uploads/" + uploaded_file.filename, target_size=(300, 300))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)
            prediction = None
            if classes[0] > 0.5:
                print(uploaded_file.filename + " is a human")
                prediction = Prediction(uploaded_file.filename, "human")
            else:
                print(uploaded_file.filename + " is a horse")
                prediction = Prediction(uploaded_file.filename, "horse")
            predictions.append(prediction)
    predictions_json = get_predictions_json(predictions)
    predictions_str = json.dumps(predictions_json)
    session['predictions'] = predictions_str
    return redirect(url_for('predict', data=predictions_str))


@app.route('/sign_s3/')
def sign_s3():
    S3_BUCKET = "horses-or-humans"# os.environ.get('S3_BUCKET')

    file_name = request.args.get('file_name')
    file_type = request.args.get('file_type')

    s3 = boto3.client('s3')

    presigned_post = s3.generate_presigned_post(
        Bucket=S3_BUCKET,
        Key=file_name,
        Fields={"acl": "public-read", "Content-Type": file_type},
        Conditions=[
            {"acl": "public-read"},
            {"Content-Type": file_type}
        ],
        ExpiresIn=3600
    )

    return json.dumps({
        'data': presigned_post,
        'url': 'https://%s.s3.amazonaws.com/%s' % (S3_BUCKET, file_name)
    })
