
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from tqdm import tqdm
from backbone import Backbone
from tensorflow import keras

BASE_PATH = "../../dataset/crawl/flickr/train/"

if __name__ == '__main__':
    dataset = pd.read_csv("../../dataset/crawl/flickr/classes/train_flickr_cells_65.csv")
    model = Backbone((240, 240))
    latest = tf.train.latest_checkpoint("../../checkpoints/chckp_99_1007")
    model.build([99])
    model.load_weights(latest)
    max_conf_correct = []
    max_conf_wrong = []
    for i in range(0,99):
        max_conf_correct.append({
            "img_path": "",
            "confidence": 0
        })
        max_conf_wrong.append({
            "img_path": "",
            "confidence": 0
        })

    with tqdm(total = dataset.shape[0]) as pbar:
        for _, row in dataset.iterrows():
            pbar.update(1)
            path = os.path.join(BASE_PATH, row["img_path"])

            background = Image.open(path).convert('RGB')
            background = background.resize((240, 240))

            img_array_p = np.array(background)
            #img_array_p = np.expand_dims(img_array_p, 0) / 255.

            res = model.get_model().predict(img_array_p)
            print("res_max:"+str(res.max()) + "; res_argmax" + str(res.argmax()))
            if res.argmax() == row["lvl_2"]:
                if res.max() > max_conf_correct[res.argmax()]["confidence"]:
                    max_conf_correct[res.argmax()]["confidence"] = res.max()
                    max_conf_correct[res.argmax()]["img_path"] = row["img_path"]
            else:
                max_conf_wrong[res.argmax()]["confidence"] = res.max()
                max_conf_wrong[res.argmax()]["img_path"] = row["img_path"]

    pd.DataFrame(max_conf_correct).to_csv("conf_correct.csv")
    pd.DataFrame(max_conf_wrong).to_csv("conf_wrong.csv")