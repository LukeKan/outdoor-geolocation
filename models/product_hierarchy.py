import tensorflow as tf
import numpy as np
from keras.regularizers import l2
from tensorflow.python.keras.applications.efficientnet import EfficientNetB1, layers
from tensorflow import losses
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
from keras import backend as K


UNFROZEN_LAYERS = 50


def transpose(matrix):
    rows = len(matrix)
    columns = len(matrix[0])

    matrix_T = []
    for j in range(columns):
        row = []
        for i in range(rows):
            row.append(matrix[i][j])
        matrix_T.append(row)

    return matrix_T


class Product_Branch(tf.keras.layers.Layer):
    def __init__(self, lvl_matrices, out_classes):
        super(Product_Branch, self).__init__()
        self.lvl_matrices = lvl_matrices
        self.out_classes = out_classes
        self.curr_product = None

    def call(self, inputs):
        self.curr_product = tf.ones([32, self.out_classes])
        for lvl in range(len(self.lvl_matrices)):
            curr_matrix = np.array(self.lvl_matrices[lvl]).astype(np.float32)
            curr_matrix_tensor = tf.constant(curr_matrix)
            self.curr_product *= tf.matmul(inputs[lvl], curr_matrix_tensor)
        return self.curr_product


class Backbone:

    def __init__(self, scale, hierarchy_tree, bs=32, alpha=0.1):
        self.bs = bs
        self.scale = scale
        self.model = None
        self.rows_size = []
        self.out_classes = None
        self.normalizing_factors = []
        self.adding_factors=[]
        self.lvl_matrices = self._build_hierarchy_matrices(hierarchy_tree, alpha)
        self.alpha = alpha
        self.efficient_net = None
        # tf.compat.v1.enable_eager_execution()
        """config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)"""



    """
    Build a list of matrices for each level of the hierarchy:
        - every element of such an array is a matrix of N columns and i rows ( N=leave cells, i = element of the 
        corresponding level);
        - every element of the matrix is a binary value which is 1 whether the corresponding cell of the level is in the
        branch that connects the current leaf to the root.

    Procedure:
        1) build the branches from the leaves upward and place the intermediate cell indices inside the curr_branch
        list and then append it inside the hierarchy_matrix list (now hierarchy_matrix is a matrix)
        2) count the number of cells for each level and place it inside the rows_size list
        3) build the binary matrices for the output
    """

    def _build_hierarchy_matrices(self, hierarchy_tree, alpha):
        hierarchy_tree = hierarchy_tree.sort_values(by=["lvl", "node_hex_id"], axis=0)
        bottom_lvl = hierarchy_tree["lvl"].max()
        leaves_cells = hierarchy_tree[
            hierarchy_tree["children_hex_ids"].str.len().eq(2)]  # 2 means that children_hex_id is empty : []
        self.out_classes = leaves_cells.shape[0]
        hierarchy_matrix = []

        curr_branch = []

        for _, cell in leaves_cells.iterrows():
            current_cell = cell
            lvl = current_cell["lvl"]
            norm_factor = 0
            for i in range(lvl + 1):
                norm_factor += pow(alpha, bottom_lvl - i - 1)
            self.normalizing_factors.insert(0, 1 / norm_factor)
            while lvl >= 0:
                curr_lvl_cells = hierarchy_tree[hierarchy_tree["lvl"] == lvl].reset_index()
                father_cell = curr_lvl_cells[
                    curr_lvl_cells["children_hex_ids"].str
                        .contains(current_cell["node_hex_id"]) |
                    curr_lvl_cells["node_hex_id"].str
                        .match(current_cell["node_hex_id"])
                    ]
                curr_branch.insert(0, father_cell.index.to_list()[0])
                father_cell = father_cell.iloc[0]
                current_cell = father_cell
                lvl -= 1
            hierarchy_matrix.append(curr_branch)
            curr_branch = []
        self.hierarchy_matrix = hierarchy_matrix
        self.normalizing_factors = np.array(self.normalizing_factors).astype(np.float32)

        rows_size = []
        for lvl in range(bottom_lvl, -1, -1):
            rows_size.insert(0, hierarchy_tree[hierarchy_tree["lvl"] == lvl].shape[0])

        one_hot_matrices = []
        epsilon = 1e-2
        for i in range(bottom_lvl + 1):
            one_hot_matrices.append([])
        for branch in hierarchy_matrix:
            for i in range(len(rows_size)):
                binary_array = [epsilon] * rows_size[i]

                if i < len(branch):
                    binary_array[branch[i]] = 1.
                one_hot_matrices[i].append(binary_array)

        for lvl in range(0, bottom_lvl+1):
            adding_array = [0.] * self.out_classes
            for i in range(self.out_classes):
                if lvl >= len(self.hierarchy_matrix[i]):
                    adding_array[i] = 1.
            self.adding_factors.append(adding_array)



        """
        Transpose matrices
        """
        for i in range(len(one_hot_matrices)):
            one_hot_matrices[i] = transpose(one_hot_matrices[i])
        self.rows_size = rows_size
        return one_hot_matrices

    def product_hierarchy(self, inputs):
        curr_product = tf.ones([32, self.out_classes])
        for lvl in range(len(self.lvl_matrices)):
            curr_matrix = np.array(self.lvl_matrices[lvl]).astype(np.float32)
            curr_matrix_tensor = tf.constant(curr_matrix)
            curr_product *= tf.matmul(inputs[lvl], curr_matrix_tensor)
        return curr_product

    def build(self):
        input_shape = self.scale + (3,)
        self.efficient_net = EfficientNetB1(include_top=False, weights='imagenet', input_shape=input_shape,
                                            drop_connect_rate=0.4)
        self.efficient_net.trainable = False
        core = self.efficient_net.output

        core = tf.keras.layers.GlobalMaxPooling2D(name="gmp")(core)
        core = tf.keras.layers.Dense(1280, activation='relu')(core)
        cell_levels = [None] * len(self.rows_size)
        hierarchy_connection = [0] * len(self.rows_size)
        add_connection = [0] * len(self.rows_size)
        for i in range(len(self.rows_size)):
            cell_levels[i] = tf.keras.layers.Dense(self.rows_size[i], name="cells_lvl_" + str(i), activation='softmax')(
                core)
            hierarchy_connection[i] = tf.constant(self.lvl_matrices[i])
            add_connection[i] = tf.constant(tf.ones(shape=[32, 72]))
            cell_levels[i] = tf.keras.layers.Lambda(lambda x:
                                                    K.dot(x[0], x[1]))([cell_levels[i], hierarchy_connection[i]])
            cell_levels[i] = tf.keras.layers.add([cell_levels[i], add_connection[i]])

            # merge_layer.trainable = False
            # cell_levels[i] = merge_layer
        # output = tf.keras.layers.concatenate(cell_levels, axis=1)
        # output = tf.keras.layers.Dense(self.out_classes, activation="softmax")(output)
        """leaves = []
        for branch in self.hierarchy_matrix:
            lvl = 0
            curr_leaf = []
            for i in branch:
                branching_layer = tf.keras.layers.Lambda(lambda x: x[:, i:i+1])(cell_levels[lvl])
                lvl += 1
                curr_leaf.append(branching_layer)
            if len(curr_leaf) > 1:
                leaves.append(tf.keras.layers.Multiply()(curr_leaf))
            else:
                leaves.append(curr_leaf[0])
        output = tf.concat(leaves, axis=1)"""
        output = tf.keras.layers.Multiply()(cell_levels)
        output = tf.keras.layers.Dense(self.out_classes, activation="softmax")(output)
        self.model = tf.keras.Model(inputs=self.efficient_net.input,
                                    outputs=[output])

        self.model.summary()

    def compile(self, optimizer=Adam(learning_rate=1e-2)):
        loss = losses.SparseCategoricalCrossentropy()
        metrics = ['accuracy']
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, data_gen, checkpoint_path):

        callbacks = []
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        adaptive_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                           patience=3, min_lr=0.000001)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        # callbacks.append(es_callback)
        callbacks.append(adaptive_lr)
        # callbacks.append(cp_callback)
        dataset_size = data_gen.train_and_valid_size()
        train_steps = int(dataset_size[0] / self.bs)
        valid_steps = int(dataset_size[1] / self.bs)
        self.model.fit(x=data_gen.generate_batch(train=True), validation_data=data_gen.generate_batch(train=False),
                       epochs=5, steps_per_epoch=train_steps, validation_steps=valid_steps,
                       callbacks=callbacks)
        self.model.save_weights(checkpoint_path + "ckpt_1.ckpt")

    def save_weights(self, path):
        self.model.save(path)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def get_model(self):
        return self.model

    def unfreeze(self):
        for layer in self.efficient_net.layers[0:-1]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
