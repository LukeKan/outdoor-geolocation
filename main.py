import os
import matplotlib.pyplot as plt
import cv2

import s2sphere
import pandas as pd
bs = 64 # Batch size
num_classes = 3
class_list = ['Rock',
              'Paper',
              'Scissors']
img_resize = [128, 128]

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    #model = Backbone()
    train = pd.read_csv("dataset/google-landmark-v2/data/train.csv")

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

    show_sample(get_paths(0))
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
