import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

from models.exp_log_sum_hierarchy import Backbone
from dataGenerator import DataGenerator

bs = 32  # Batch size
img_resize = (240, 240)
classes_size = [72]
DATA_PATH = "dataset/crawl/flickr"


def main():
    # Use a breakpoint in the code line below to debug your script.
    BASE_FOLDER = os.path.abspath(os.path.join(os.getcwd(), DATA_PATH))
    """data_generator = DataGenerator(os.path.join(BASE_FOLDER, "train_flickr_cells_65.csv"),
                                   os.path.join(BASE_FOLDER, "train"), bs, classes_size, SAMPLES_NUMBER)"""
    hierarchy = pd.read_csv(os.path.join(os.path.join(BASE_FOLDER, "hierarchy"), "cells_hierarchy_72.csv"))
    model = Backbone(scale=img_resize, bs=bs, hierarchy_tree=hierarchy, alpha=0.5)
    model.build()
    model.compile()
    # model.train(data_generator, "")


if __name__ == '__main__':
    main()
