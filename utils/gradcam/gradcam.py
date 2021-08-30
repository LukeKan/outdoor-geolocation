import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

# Display
from IPython.display import display

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm

from backbone import Backbone
BASE_PATH = "../../dataset/crawl/flickr/train/"

class GradCAM:

    def __init__(self, model, img_size):
        self.model = model
        self.img_size = img_size
        self.last_conv_layer_name = "top_conv"

        # The local path to our target image


    def get_img_array(self, img_path, size):
        # `img` is a PIL image of size 299x299
        img = keras.preprocessing.image.load_img(img_path, target_size=size)
        # `array` is a float32 Numpy array of shape (299, 299, 3)
        array = keras.preprocessing.image.img_to_array(img)
        # We add a dimension to transform our array into a "batch"
        # of size (1, 299, 299, 3)
        array = np.expand_dims(array, axis=0)
        return array

    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def save_and_display_gradcam(self, img_path, cam_path="cam.jpg", alpha=0.4):
        # Load the original image
        img = keras.preprocessing.image.load_img(img_path)
        img = keras.preprocessing.image.img_to_array(img)
        img_array = self.get_img_array(img_path, size=self.img_size)

        # Rescale heatmap to a range 0-255
        heatmap = self.make_gradcam_heatmap(img_array, self.model.get_model(), self.last_conv_layer_name)
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path)


if __name__ == '__main__':
    model = Backbone((240, 240))
    latest = tf.train.latest_checkpoint("../../checkpoints/chckp_99_1007")
    model.build([99])
    model.load_weights(latest)

    gradcam = GradCAM(model, (240, 240))
    max_conf_correct = pd.read_csv("conf_correct.csv")
    max_conf_wrong = pd.read_csv("conf_wrong.csv")

    with tqdm(total = max_conf_correct.shape[0]) as pbarC:
        for _, row in max_conf_correct.iterrows():
            pbarC.update(1)
            gradcam.save_and_display_gradcam(img_path=os.path.join(BASE_PATH,row["img_path"]),cam_path="correct/"+row["img_path"])
    with tqdm(total = max_conf_wrong.shape[0]) as pbarW:
        for _, row in max_conf_wrong.iterrows():
            pbarW.update(1)
            gradcam.save_and_display_gradcam(img_path=os.path.join(BASE_PATH,row["img_path"]),cam_path="wrong/"+row["img_path"])