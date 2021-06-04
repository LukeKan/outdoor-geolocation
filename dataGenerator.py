from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import random
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFile
from sklearn.model_selection import train_test_split


class DataGenerator:

    SIZE = (224, 224)

    def __init__(self, csv_train_path, image_dir, target_list):
        self.train_dict = pd.read_csv(csv_train_path)
        self.target_list=target_list
        self.IMAGE_DIR = image_dir
        self.train_data, self.valid_data, self.train_target, self.valid_target = [],[],[],[]

    def get_dataset(self):
        train_data, valid_data, train_target, valid_target = train_test_split(self.train_dict["url"],
                                                                                                  self.train_dict["tag"],
                                                                                                  test_size=0.2,
                                                                                                  random_state=42)
        pd_train = pd.DataFrame(data={'id': train_data, 'label': train_target}).sample(frac=1).reset_index(drop=True)
        #print(pd_train.head(10))
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
        train_ds = train_datagen.flow_from_dataframe(dataframe=pd_train, directory=self.IMAGE_DIR, subset="training",
                                                     x_col='id', shuffle=True, class_mode="categorical",
                                                     y_col='label', target_size=self.SIZE)
        pd_valid = pd.DataFrame(data={'id': valid_data, 'label': valid_target}).sample(frac=1).reset_index(drop=True)
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        valid_ds = valid_datagen.flow_from_dataframe(dataframe=pd_valid, directory=self.IMAGE_DIR, subset="training",
                                                     x_col='id', shuffle=True, class_mode="categorical",
                                                     y_col='label', target_size=self.SIZE)

        return train_ds, valid_ds

    def rotate(self, img):
        # random selection between rotations
        rotation = random.randrange(-90, 90)
        rotated_img = img.rotate(rotation)
        return rotated_img

    def flip(self, img):
        flip = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM])
        rotated_img = img.transpose(flip)
        return rotated_img

    def change_brightness(self, img):
        enhancer = ImageEnhance.Brightness(img)

        factor = 1  # gives original image
        enhanced_img = enhancer.enhance(factor)
        return enhanced_img

    def mask(self, img):

        while True:
            y_top = np.random.randint(0, self.SIZE[1])  # row index
            x_top = np.random.randint(0, self.SIZE[0])  # column index

            height_max = self.SIZE[1] - y_top
            width_max = self.SIZE[0] - x_top

            if height_max >= 10 and width_max >= 10:
                break

        height = np.random.randint(5, height_max)
        width = np.random.randint(5, width_max)

        c = np.random.uniform(0, 1)

        img[:, y_top:y_top + height, x_top:x_top + width, :] = c

        return img

    def get_image(self, image_id, data_augmentation=False):

        path = self.IMAGE_DIR + self.train_dict[image_id].split('/')[4]

        background = Image.open(path).convert('RGB')
        background = background.resize((self.SIZE[1], self.SIZE[0]))

        if data_augmentation:
            background = self.mask(background)
            data_agumentations = [self.rotate, self.flip, self.change_brightness]
            background = random.choice(data_agumentations)

        img_array = np.array(background)
        img_array = np.expand_dims(img_array, 0)
        img_array = img_array / 255.

        return img_array

    def generate_batch(self, bs, train=True):
        while True:
            data_augmentation = np.random.choice((True, False), bs, p=[0.5, 0.5])
            batch_images = []
            target_index = 0
            y1 = np.empty(bs, dtype=int)
            #y2 = np.empty(bs, dtype=int)
            #y3 = np.empty(bs, dtype=int)
            #y4 = np.empty(bs, dtype=int)

            # TODO : 1. Sample city, 2. sample images, 3. get image path, 4. get image
"""
            city_batch = np.random.choice(a=cities, size=bs, replace=True)
            city_frequency = Counter(city_batch)
            for city, freq in city_frequency.items():
                ids = np.random.choice(a=images_per_city[int(city)], size=freq, replace=True)
                for i, id in enumerate(ids):
                    hotel = int(image_hotel_dict[id])
                    batch_images.extend(self.get_image(id, data_augmentation[i]))
                    y1[target_index] = int(self.hotel_info_dict[hotel][1])
                    target_index += 1
"""
            #X = np.array(batch_images)
            #y1 = np.array(y1)
            #y2 = np.array(y2)
            #y3 = np.array(y3)
            #y4 = np.array(y4)

            #yield X, [y1]#, y2, y3, y4]

def get_target(self):
        target_dict = defaultdict(list)
        test_images = {row[0]: row[1] for _, row in
                       pd.read_csv(self.IMAGE_DIR + "train.csv").iterrows()}  # ImageId : HotelId

        for image_id, hotel_id in test_images.items():
            info = self.hotel_info_dict[hotel_id]
            int_info = [int(i) for i in info]

            target_dict[str(image_id)] = [int_info[1], int_info[2], int_info[3],
                                          int_info[0]]  # country, city, subregion, chain

        return target_dict