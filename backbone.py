import tensorflow as tf
from keras.regularizers import l2
from tensorflow.python.keras.applications.efficientnet import EfficientNetB1, layers
from tensorflow import losses
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession


UNFROZEN_LAYERS = 50
class Backbone:

    def __init__(self, scale, bs=32):
        self.bs = bs
        self.scale = scale
        self.model = None
        self.num_classes = 0

    def build(self, out_lvls_size):
        input_shape = self.scale + (3,)
        self.efficient_net = EfficientNetB1(include_top=False, weights='imagenet', input_shape=input_shape,
                                            drop_connect_rate=0.4)
        self.efficient_net.trainable = False
        """for layer in efficient_net.layers[:-UNFROZEN_LAYERS]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True"""
        core = self.efficient_net.output

        core = tf.keras.layers.GlobalMaxPooling2D(name="gap")(core)
        core = tf.keras.layers.Dense(1280, kernel_regularizer=l2(0.0001))(core)
        out_lvl2 = tf.keras.layers.Dense(out_lvls_size[0], name="cell_no", activation='softmax')(core)
        self.model = tf.keras.Model(inputs=self.efficient_net.input,
                                    outputs=[out_lvl2])

        self.model.summary()
        self.num_classes = out_lvls_size[0]

    def compile(self, optimizer=Adam(learning_rate=1e-4)):
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
        #callbacks.append(es_callback)
        callbacks.append(adaptive_lr)
        #callbacks.append(cp_callback)
        dataset_size = data_gen.train_and_valid_size()
        train_steps = int(dataset_size[0] / self.bs)
        valid_steps = int(dataset_size[1] / self.bs)
        self.model.fit(x=data_gen.generate_batch(train=True), validation_data=data_gen.generate_batch(train=False),
                       epochs=5, steps_per_epoch=train_steps, validation_steps=valid_steps,
                       callbacks=callbacks)
        self.unfreeze()
        self.model.fit(x=data_gen.generate_batch(train=True), validation_data=data_gen.generate_batch(train=False), epochs=5, steps_per_epoch=train_steps, validation_steps=valid_steps, callbacks=callbacks)
        self.model.save_weights(checkpoint_path + "ckpt_1.ckpt")

    def save_weights(self, path):
        self.model.save(path)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def get_model(self):
        return self.model

    def unfreeze(self):
        for layer in self.efficient_net.layers[:-UNFROZEN_LAYERS]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    def get_output_classes(self):
        return self.num_classes
