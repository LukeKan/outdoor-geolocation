import os

from backbone import Backbone
from dataGenerator import DataGenerator

bs = 128  # Batch size

num_classes = 18
tag_list = ['city'
                , 'building'
                , 'countryside'
                , 'landscape'
                , 'mountain'
                , 'woods'
                , 'avenue'
                , 'highway'
                , 'street'
                , 'town'
                , 'monument'
                , 'glacier'
                , 'parkway'
                , 'lane'
                , 'roadway'
                , 'neighborhood'
                , 'thoroughfare'
                , 'motorway'
                ]
img_resize = [256, 256]

"""
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
"""
def main():
    # Use a breakpoint in the code line below to debug your script.
    BASE_FOLDER = os.path.abspath(os.path.join(os.getcwd(), "data"))
    data_generator = DataGenerator(os.path.join(BASE_FOLDER,"train_flickr_clean.csv"), os.path.join(BASE_FOLDER,"train"),tag_list)
    train_ds, valid_ds = data_generator.get_dataset()


    model = Backbone(classes=tag_list, bs=32)
    model.build(num_classes)
    model.compile()
    model.train(train_ds, valid_ds)
    """"
    show_sample(get_paths(0))
    """

if __name__ == '__main__':
    main()

