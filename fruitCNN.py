from scipy.misc import imsave
import numpy as np
import ImageLoader
import time
from keras.applications import vgg16
from keras import backend as K


VGG_MEAN = [103.939, 116.779, 123.68]

class fruitCNN:
    """
    Loads images for training from.
    """

    def __init__(self, vgg16_npy_path):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1')
        self.train_data = np.empty((0, 224, 224, 3))

    def addToDataset(self, path):
        imLoader = ImageLoader(path)
        if (self.train_data.shape[0] == 0):
            self.train_data = imLoader.train_set
        else:
            self.train_data = np.concatenate((self.train_data, imLoader.train_set), 0)
        imLoader.closeFile()

    def makeTrainTestBatches(self, batch_size=32, useImageTransforms=True):