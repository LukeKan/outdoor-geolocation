import os
from glob import glob

import tensorflow as tf
import pandas as pd
import numpy as np
import s2sphere
from keras import layers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0
from tensorflow import losses
from tensorflow.python.data import Dataset
from tensorflow.python.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tqdm import tqdm

class Backbone:
    targets = []
    data_gen = {}
    model = {}
    scale = (224, 224)
    sampling_rate = 0.02  # fraction of images from each landmark_id
    random_state = 17  # for reproducibility
    bs = 32

    def __init__(self, classes, bs=32):
        self.targets = classes
        self.bs = bs
        #train_gen, validation_gen = self._imgDataGen(dataset_dir=dataset_path, bs=bs)
        #self.train_dataset, self.valid_dataset = self._get_datasets(train_gen, validation_gen)

    def path_to_tensor(self, img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=self.scale)

        # convert PIL.Image.Image type to 3D tensor with shape (scale, 3)
        x = image.img_to_array(img)

        # convert 3D tensor to 4D tensor with shape (1, scale, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def paths_to_tensor(self, img_paths):
        list_of_tensors = [self.path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)

    def load_dataset(self, path, train_sample):
        file_out = sorted(glob(path + '/*/*/*/*'))
        file_out = np.array([s.replace("\\", "/") for s in file_out])

        label_out = pd.Series(name="landmark_id")

        for file in file_out:
            filebase = os.path.basename(file)
            name = os.path.splitext(filebase)[0]
            temp = train_sample.landmark_id[train_sample["id"].values == name]
            label_out = label_out.append(temp)
            print(name + " processed.")

        label_out = np.array(pd.get_dummies(label_out))

        return file_out, label_out



    def get_dataset(self, train_df):
        train_path = 'dataset/google-landmark-v2/data/train'
        valid_path = 'dataset/google-landmark-v2/data/train'
        #test_path = './test_images/'
        train_file, train_target = self.load_dataset(train_path, train_df)
        valid_file, valid_target = train_file, train_target
        #test_file, test_target = self.load_dataset(test_path, train_sample)
        train_tensors = self.paths_to_tensor(train_file).astype('float32') / 255
        valid_tensors = train_tensors
        #test_tensors = self.paths_to_tensor(test_file).astype('float32') / 255
        return train_tensors, train_target, valid_tensors, valid_target


    def build(self, final_size):
        input_shape = self.scale + (3,)
        self.model = Sequential()
        efficient_net = EfficientNetB0(include_top=True, weights='imagenet', input_shape=input_shape)
        for layer in efficient_net.layers:
            layer.trainable = False
        self.model.add(efficient_net)

        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=final_size, activation='softmax'))

        self.model.summary()

    def compile(self, optimizer=Adam(learning_rate=1e-3)):
        loss = losses.CategoricalCrossentropy()
        metrics = ['accuracy']
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_tensors, train_target, valid_tensors, valid_target):
        self.model.fit(train_tensors, train_target, validation_data=(valid_tensors, valid_target), epochs=60,
                       verbose=1, batch_size=self.bs)
