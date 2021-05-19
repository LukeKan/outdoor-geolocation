import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from backbone import Backbone

bs = 128  # Batch size

num_classes = 203093
class_list = range(0, 203093)
img_resize = [256, 256]


def get_paths(index_location):
    index = os.listdir('dataset/google-landmark-v2/data/train/')
    paths = []
    a = index_location
    for b in index:
        for c in index:
            try:
                paths.extend([f"dataset/google-landmark-v2/data/train/{a}/{b}/{c}/" + x for x in
                              os.listdir(f"dataset/google-landmark-v2/data/train/{a}/{b}/{c}")])
            except:
                pass

    return paths


def show_sample(pathes):
    plt.rcParams["axes.grid"] = False
    f, axarr = plt.subplots(3, 3, figsize=(20, 20))
    axarr[0, 0].imshow(cv2.imread(pathes[0]))
    axarr[0, 1].imshow(cv2.imread(pathes[1]))
    axarr[0, 2].imshow(cv2.imread(pathes[2]))
    axarr[1, 0].imshow(cv2.imread(pathes[3]))
    axarr[1, 1].imshow(cv2.imread(pathes[4]))
    axarr[1, 2].imshow(cv2.imread(pathes[5]))
    axarr[2, 0].imshow(cv2.imread(pathes[6]))
    axarr[2, 1].imshow(cv2.imread(pathes[7]))
    axarr[2, 2].imshow(cv2.imread(pathes[8]))
    plt.show()

def main():
    # Use a breakpoint in the code line below to debug your script.
    model = Backbone(classes=class_list, bs=128)
    train_df = pd.read_csv('dataset/google-landmark-v2/data/train.csv')
    train_tensors, train_target, valid_tensors, valid_target = model.get_dataset(train_df)
    model.build()
    model.compile()
    model.train(train_tensors, train_target, valid_tensors, valid_target)

    show_sample(get_paths(0))


if __name__ == '__main__':
    main()

