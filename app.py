
from flask import Flask, request, redirect, jsonify, render_template
from flask_cors import CORS
import os
import uuid
from gevent.pywsgi import WSGIServer
from chexnet.chexnet import Xray
from util import base64_to_pil, np_to_base64, base64_to_bytes
import numpy as np
import torch
iron['BUCKET_NAME'] if 'BUCKET_NAME' in os.environ else 'covid'

app = Flask(__name__)
CORS(app)
np.set_printoptions(suppress=True)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = base64_to_pil(request.json)
    except Exception:
        return jsonify(error="expected string or bytes-like object"), 422

    rgb_image_np = np.array(img.convert('RGB'))
    stds = rgb_image_np.std(axis=(0, 1))
    means = rgb_image_np.mean(axis=(0, 1))
    if 35 < np.average(stds) < 95 and 60 < np.average(means) < 180:
        img_result = x_ray.predict(img)
        condition_similarity_rate = []
        if img_result['condition rate'] == []:
            return jsonify(
                result='NOT DETECTED',
                error='Invalid image'
            ), 200
        else:
            for name, prob in img_result['condition rate']:
                condition_similarity_rate.append({'y': round(float(prob), 3), 'name': name})
            print(condition_similarity_rate)
            return jsonify(
                condition_similarity_rate=condition_similarity_rate
            ), 200
    else:
        file_name = "NOT_DETECTED/%s.jpg" % str(uuid.uuid4())
        return jsonify(
            result='NOT DETECTED',
            error='Invalid image'
        ), 200


if __name__ == '__main__':
    x_ray = Xray()
    http_server = WSGIServer(('0.0.0.0', 5000), app, log=None)
    http_server.serve_forever()
