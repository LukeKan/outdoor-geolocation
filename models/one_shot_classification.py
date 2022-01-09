import os
from glob import glob

import tensorflow as tf
import pandas as pd
import numpy as np
from keras import layers
from keras.regularizers import l2
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
        core = efficient_net.output

        core = tf.keras.layers.GlobalMaxPooling2D(name="gap")(core)
        #core = tf.keras.layers.Dense(1280, activation='relu', kernel_regularizer=l2(0.00001))(core)
        out_lvl2 = tf.keras.layers.Dense(out_lvls_size[0], name="cell_no", activation='softmax')(core)
        self.model = tf.keras.Model(inputs=efficient_net.input,
                                    outputs=[out_lvl2])

        self.model.summary()

    def compile(self, optimizer=Adam(learning_rate=1e-2)):
        loss = losses.SparseCategoricalCrossentropy()
        metrics = ['accuracy']
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, data_gen, checkpoint_path):
        callbacks = []
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        adaptive_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                           patience=3, min_lr=0.000001)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        callbacks.append(es_callback)
        callbacks.append(adaptive_lr)
        # callbacks.append(cp_callback)
        dataset_size = data_gen.train_and_valid_size()
        train_steps = round(int(dataset_size[0] / self.bs) * 1.3)
        valid_steps = int(dataset_size[1] / self.bs)
        self.model.fit(x=data_gen.generate_batch(train=True), validation_data=data_gen.generate_batch(train=False),
                       epochs=1, steps_per_epoch=train_steps/2, validation_steps=valid_steps,
                       callbacks=callbacks)
        self.model.save_weights(checkpoint_path + "ckpt_1.ckpt")

    def save_weights(self, path):
        self.model.save(path)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def get_model(self):
        return self.model
