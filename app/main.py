#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import render_template
from flask import Flask, flash, request, redirect, url_for, json
from werkzeug.utils import secure_filename
import io
import numpy as np
from keras.models import model_from_json
from PIL import Image
import os
import scipy
import math 
from skimage.transform import resize
import tensorflow as tf
import base64

ALLOWED_EXTENSIONS = {'jpg','png','jpeg'}


app = Flask(__name__,static_url_path='/static')


model = None
graph = None


def img_to_base64_str( img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    return "data:image/png;base64,{}".format(base64.b64encode(buffered.getvalue()).decode())


def fix_image_size(tgimg,layers):
    initial_width = tgimg.shape[0]
    initial_height = tgimg.shape[1]
    new_width = int(_calculate_output_size(initial_width,0,layers) * ( 2** layers)) 
    new_height = int(_calculate_output_size(initial_height,0,layers) * ( 2** layers)) 
    tgimg = resize(tgimg,(new_width,new_height),0,preserve_range=True)
    return tgimg


def _calculate_output_size(size,level,stop_level):
    if level == stop_level :
        return size
    return _calculate_output_size(math.ceil( size / 2 ),level + 1,stop_level)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    global model
    json_file = open(os.getenv('NN_MODEL', 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.getenv('NN_WEIGHTS', 'test.h5'))
    print("Model Loaded from disk")

    model.compile(
        optimizer= "adam",
        loss= "mse")

    global graph
    graph = tf.get_default_graph() 


@app.route('/predict',methods=['POST'])
def predict():
   
    global model
    global graph


    file = request.files['file']

    if file.filename == '':
        response = app.response_class(
            response=json.dumps({"error":"No selected file"}),
            status=401,
            mimetype='application/json'
        )
        return response


    if file and allowed_file(file.filename):
        img = Image.open(io.BytesIO(file.stream.read()))
        data = np.asarray( img, dtype="int32" )[...,:3]
        old_shape = data.shape
        x = fix_image_size(data,2) / 127.5 - 1

        with graph.as_default():
            prediction= model.predict(x[np.newaxis,...])[0]

        #prediction = resize(prediction,old_shape)
        encoded_string = img_to_base64_str(Image.fromarray(np.uint8(np.clip((prediction + 1 ) * 127.5, 0, 255) )))

        response = app.response_class(
            response=json.dumps({"encoded":encoded_string}),
            status=201,
            mimetype='application/json'
        )
        return response

    response = app.response_class(
            response=json.dumps({"error":"No allowed file"}),
            status=404,
            mimetype='application/json'
        )
    return response


@app.route('/', methods=['GET'])
def upload_file():
    return render_template('index.html')


load_model()
if __name__ == '__main__':
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'O$%7kQgXKVOhT@refsbY;mQmt9lMWg')
    port = os.getenv('PORT', 9000)
    print('port=', port)
    app.run(host='0.0.0.0', debug=True, port=port)
