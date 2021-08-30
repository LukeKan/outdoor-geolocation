import os
from glob import glob

import tensorflow as tf
import pandas as pd
import numpy as np
from keras import layers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.efficientnet import EfficientNetB2, \
    EfficientNetB1
from tensorflow import losses
from tensorflow.python.data import Dataset
from tensorflow.python.keras.applications.resnet import ResNet152
from tensorflow.python.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tqdm import tqdm
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
UNFROZEN_LAYERS = 50
class Backbone:

    def __init__(self, scale, bs=32):
        self.bs = bs
        self.scale = scale
        self.model = None

    def build(self, out_lvls_size):
        input_shape = self.scale + (3,)
        efficient_net = EfficientNetB1(include_top=False, weights='imagenet', input_shape=input_shape)
        efficient_net.trainable = False
        for layer in efficient_net.layers[:-UNFROZEN_LAYERS]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        core = efficient_net.output

        core = tf.keras.layers.GlobalMaxPooling2D(name="gap")(core)
        #core = tf.keras.layers.BatchNormalization()(core)
        out_lvl2 = tf.keras.layers.Dense(out_lvls_size[0], name="cell_no", activation='softmax')(core)
        self.model = tf.keras.Model(inputs=efficient_net.input,
                                    outputs=[out_lvl2])

        self.model.summary()

    def compile(self, optimizer=Adam(learning_rate=1e-2)):
        loss = losses.SparseCategoricalCrossentropy()
        metrics = ['accuracy']
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, data_gen):
        callbacks = []
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        adaptive_lr = tf.keras.callbacks.ReduceLROnPlateau()
        callbacks.append(es_callback)
        callbacks.append(adaptive_lr)
        dataset_size = data_gen.train_and_valid_size()
        train_steps = int(dataset_size[0] / self.bs)
        valid_steps = int(dataset_size[1] / self.bs)
        self.model.fit(x=data_gen.generate_batch(train=True), validation_data=data_gen.generate_batch(train=False),
                       epochs=5, steps_per_epoch=train_steps, validation_steps=valid_steps,
                       callbacks=callbacks)
