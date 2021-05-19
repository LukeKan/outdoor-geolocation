import tf as tf
import s2sphere
from tensorflow import losses
from tensorflow.python.data import Dataset
from tensorflow.python.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


class Backbone:
    targets = []
    data_gen = {}
    model = {}
    scale = [224, 224]
    train_dataset, valid_dataset = {}

    def __init__(self, classes, dataset_path, bs=1):
        self.targets = classes
        train_gen, validation_gen = self._imgDataGen(dataset_dir=dataset_path, bs=bs)
        self.train_dataset, self.valid_dataset = self._get_datasets(train_gen, validation_gen)

    def _imgDataGen(self, dataset_dir, bs):
        train_data_gen = ImageDataGenerator(rotation_range=40,
                                            width_shift_range=10,
                                            height_shift_range=10,
                                            zoom_range=0.3,
                                            shear_range=0.4,
                                            brightness_range=(0.3, 0.7),
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='constant',
                                            validation_split=0.2,
                                            cval=0,
                                            rescale=1. / 255)
        train_gen = train_data_gen.flow_from_directory(dataset_dir,
                                                       batch_size=bs,
                                                       class_mode='categorical',
                                                       classes=self.targets,
                                                       shuffle=True,
                                                       subset='train',
                                                       target_size=self.scale,
                                                       color_mode='rgb')

        validation_gen = train_data_gen.flow_from_directory(dataset_dir,
                                                            batch_size=bs,
                                                            class_mode='categorical',
                                                            classes=self.targets,
                                                            shuffle=True,
                                                            subset='validation',
                                                            target_size=self.scale,
                                                            color_mode='rgb')
        return train_gen, validation_gen

    def _get_datasets(self, train_gen, validation_gen):

        train_dataset = Dataset.from_generator(lambda: train_gen,
                                                       output_types=(tf.float32, tf.float32),
                                                       output_shapes=(
                                                       [None, 224, 224, 3], [None, len(self.targets)]))
        train_dataset = train_dataset.repeat()

        # Validation
        # ----------
        valid_dataset = Dataset.from_generator(lambda: validation_gen,
                                                       output_types=(tf.float32, tf.float32),
                                                       output_shapes=(
                                                       [None, 224, 224, 3], [None, len(self.targets)]))

        # Repeat
        valid_dataset = valid_dataset.repeat()
        return train_dataset, valid_dataset

    def build(self):
        self.model = ResNet101V2(
            include_top=True, weights='imagenet', input_tensor=None,
            input_shape=None, pooling=None, classes=len(self.targets)
        )
        for layer in self.model.layers:
            layer.trainable = False

        self.model.summary()

    def compile(self, logdir='', optimizer=Adam(learning_rate=1e-3)):
        loss = losses.CategoricalCrossentropy()
        metrics = ['accuracy']
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self):
        self.model.fit(x=self.train_dataset, epochs=60, steps_per_epoch=40, validation_data=self.valid_dataset, validation_steps=5)
