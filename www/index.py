from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from tqdm import tqdm
from backbone import Backbone
from utils.gradcam.gradcam import GradCAM
from tensorflow import keras
from IPython.display import Image as Display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import HTML

from utils.map.MapDrawer import *

IMG_SIZE = (240, 240)
CLASSES = 89
BS = 8
BASE_PATH = "../dataset/crawl/flickr/test/"

# Geolocation model
model = Backbone(IMG_SIZE)
model.build([CLASSES])
latest = tf.train.latest_checkpoint("../checkpoints/chckp_89_16_07")
model.load_weights(latest)
dataset = pd.read_csv("../dataset/test/test_flickr.csv")
cell_reference = pd.read_csv("../dataset/crawl/flickr/cell_reference/cells_2000_20000_images_782718_0_30.csv",
                             index_col=False)

# Gradcam model
gradcam = GradCAM(model, IMG_SIZE)

# Flask app definition
app = Flask(__name__)

# Uploads Config
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


@app.route('/')
def home():
    return render_template('index.html', items=[])


@app.route('/map/<path:path>')
def load_map(path):
    return render_template('map_' + path + '.html', items=[])


@app.route('/assets/<path:path>')
def send_static(path):
    return send_from_directory('templates/assets', path)


@app.route('/test_img/<path:path>')
def send_img(path):
    return send_from_directory('test_img', path)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed')
        return predict_uploaded(filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/random', methods=['POST'])
def random_prediction():
    img = dataset.sample().iloc[0]
    path = os.path.join(BASE_PATH, img["img_path"])
    res = _predict_and_gradcam(path)
    myMap = drawCellsOnWorldMap([
        cell_reference[cell_reference["class_label"] == img["lvl_2"]].iloc[0],
        cell_reference[cell_reference["class_label"] == res.argmax()].iloc[0]
    ])
    map_name = img["img_path"].split(".")[0]
    myMap.save("templates/map_" + map_name + ".html")
    return render_template('index.html', items=["test_img/" + img["img_path"], "test_img/cam.jpg"],
                           output_label=res.argmax(), scroll='footer', map=map_name)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_uploaded(img_path):
    path = os.path.join(UPLOAD_FOLDER, img_path)
    res = _predict_and_gradcam(path)
    myMap = drawCellsOnWorldMap([cell_reference[cell_reference["class_label"] == res.argmax()].iloc[0]])
    map_name = img_path.split(".")[0]
    myMap.save("templates/map_" + map_name + ".html")
    return render_template('index.html', items=[UPLOAD_FOLDER + img_path, "test_img/cam.jpg"],
                           output_label=res.argmax(), scroll='footer', map=map_name)


def _predict_and_gradcam(path):
    background = Image.open(path).convert('RGB')
    background = background.resize((240, 240))
    grad_img = gradcam.save_and_display_gradcam(img_path=path, cam_path="test_img/cam.jpg")
    img_array_p = np.array(background)
    img_array_p = np.expand_dims(img_array_p, 0)
    return model.get_model().predict(img_array_p)


if __name__ == '__main__':
    app.run()
