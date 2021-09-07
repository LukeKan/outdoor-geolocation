import os
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import random
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFile
from sklearn.model_selection import train_test_split

SIZE = (240, 240)


class DataGenerator:
    SIZE = (240, 240)

    def __init__(self, csv_train_path, image_dir, bs, classes_size, samples_number):
        self.bs = bs
        np.random.seed(42)
        self.train_dict = pd.read_csv(csv_train_path).sample(frac=1).reset_index(drop=True)
        self.samples_number = samples_number
        self.classes_size = classes_size
        #self._extract_balanced_samples()
        """for _, row in self.train_dict.iterrows():
            elems = row["targets"][1:len(row["targets"]) - 1].split(",")
            elems = map(int, elems)
            row["targets"] = list(elems)"""

        self.IMAGE_DIR = image_dir
        self.train_dataset, self.valid_dataset = np.split(self.train_dict, [int(.9*len(self.train_dict))])

    def _extract_balanced_samples(self):
        sample_per_class = int(self.samples_number / self.classes_size[0])
        extracted_data = pd.DataFrame()
        for i in range(0, self.classes_size[0]):
            extracted_data = extracted_data.append(self.train_dict[self.train_dict["lvl_2"] == i][:sample_per_class])
        if extracted_data.shape[0] < self.samples_number:
            extracted_data = extracted_data.append(self.train_dict.sample(frac=1)[:self.samples_number - extracted_data.shape[0]])
            extracted_data = extracted_data.drop_duplicates()
        self.train_dict = extracted_data.sample(frac=1)

    def train_and_valid_size(self):
        return [len(self.train_dataset), len(self.valid_dataset)]

    def set_classes_size(self, classes_size):
        self.classes_size = classes_size

    def get_classes_size(self):
        return self.classes_size

    def rotate(self, img):
        # random selection between rotations
        rotated_img = tf.image.rot90(img, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        return rotated_img

    def flip(self, img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    def change_brightness(self, img):
        enhanced_img = tf.image.random_brightness(img, max_delta=0.5)
        return enhanced_img

    def zoom(self, img):
        # Generate 20 crop settings, ranging from a 1% to 20% crop.
        scales = list(np.arange(0.8, 1.0, 0.01))
        boxes = np.zeros((len(scales), 4))

        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]

        def random_crop(img):
            # Create different crops for an image
            crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(32, 32))
            # Return a random crop
            return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

        # Only apply cropping 50% of the time
        return tf.cond(choice < 0.5, lambda: img, lambda: random_crop(img))

    def color_augmentation(self, img):
        img = tf.image.random_hue(img, 0.08)
        img = self.saturate(img)
        img = self.change_brightness(img)
        img = self.contrast(img)
        return img

    def contrast(self, img):
        return tf.image.random_contrast(img, 0.5, 2.0)

    def saturate(self, img):
        return tf.image.random_saturation(img, 0.6, 1.6)

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

        path = os.path.join(self.IMAGE_DIR , self.train_dict.iloc[image_id]["img_path"])
        try:
            background = Image.open(path).convert('RGB')
        except:
            path = os.path.join("E:/dataset/train_1", self.train_dict.iloc[image_id]["img_path"])
            try:
                background = Image.open(path).convert('RGB')
            except:
                path = path = os.path.join("E:/dataset/train_2", self.train_dict.iloc[image_id]["img_path"])
                try:
                    background = Image.open(path).convert('RGB')
                except:
                    path = os.path.join("E:/dataset/train_3", self.train_dict.iloc[image_id]["img_path"])
                    background = Image.open(path).convert('RGB')
        background = background.resize((self.SIZE[1], self.SIZE[0]))

        img_array = np.array(background)
        img_array = np.expand_dims(img_array, 0)
        #img_array = img_array / 255.
        """if data_augmentation:
            background = self.mask(background)
            data_agumentations = [self.rotate, self.flip, self.color_augmentation]
            img_array = random.choice(data_agumentations)(img_array)"""

        return img_array

    def generate_batch(self,  train=True):
        while True:
            data_augmentation = np.random.choice((True, False), self.bs, p=[0.5, 0.5])
            batch_images = []
            target_index = 0
            y1 = np.empty(self.bs, dtype=int)
            #y2 = np.empty(self.bs, dtype=int)
            #y3 = np.empty(self.bs, dtype=int)
            #y4 = np.empty(self.bs, dtype=int)

            if train:
                ids = np.random.randint(low=0, high=len(self.train_dataset)-1, size=self.bs)
                for i, id in enumerate(ids):
                    batch_images.extend(self.get_image(id, data_augmentation[i]))
                    y1[target_index] = self.train_dataset.iloc[id]["0_30"]
                    #y2[target_index] = self.train_dataset.iloc[id]["lvl_3"]
                    #y3[target_index] = self.train_dataset.iloc[id]["lvl_4"]
                    #y4[target_index] = self.train_dataset.iloc[id]["lvl_5"]
                    target_index += 1
            else:
                ids = np.random.randint(low=0, high=len(self.valid_dataset)-1, size=self.bs)
                for i, id in enumerate(ids):
                    batch_images.extend(self.get_image(id, data_augmentation[i]))
                    y1[target_index] = self.valid_dataset.iloc[id]["0_30"]
                    #y2[target_index] = self.valid_dataset.iloc[id]["lvl_3"]
                    #y3[target_index] = self.valid_dataset.iloc[id]["lvl_4"]
                    #y4[target_index] = self.valid_dataset.iloc[id]["lvl_5"]
                    target_index += 1

            X = np.array(batch_images)
            y1 = np.array(y1)
            #y2 = np.array(y2)
            #y3 = np.array(y3)
            #y4 = np.array(y4)

            yield X, y1#, y2, y3, y4]
