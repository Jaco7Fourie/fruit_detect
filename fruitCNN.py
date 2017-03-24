from scipy.misc import imsave
import numpy as np
import ImageLoader
import time
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras import backend as K


VGG_MEAN = [103.939, 116.779, 123.68]

class fruitCNN:
    """
    Loads images for training from.
    """

    def __init__(self, vgg16_npy_path, classes):
        """
        initialise with number of classes and path to weights file
        :param vgg16_npy_path:
        :param classes:
        """
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1')
        self.data = np.empty((0, 224, 224, 3))
        self.labels = np.empty((0, classes))

    def addToDataset(self, path, label):
        """
        adds a set of imge matches to train_data based on a log file of labelled matches
        :param path: the path to the matches log
        :param label: an integer np_array on-hot vector representing the label
        :return: None
        """
        imLoader = ImageLoader(path)
        if (self.data.shape[0] == 0):
            self.data = imLoader.train_set
            self.labels = label
        else:
            self.data = np.concatenate((self.data, imLoader.train_set), 0)
            self.labels = np.concatenate((self.labels, label), 0)
        imLoader.closeFile()

    def makeTrainTestBatches(self, b_size=32):
        """
        returns a generator used with  model.fit_generator to generate batches of transformed images
        :param b_size: the batch size to feed into the network
        :return: a batch generator compatible with model.fit_generator
        """
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.0,
            height_shift_range=0.0,
            shear_range=0.2,
            zoom_range=0.0,
            horizontal_flip=True,
            fill_mode='nearest')
        return datagen.flow(self.data, self.labels, batch_size=b_size)

    def buildModel(self):
        model = applications.VGG16(weights='imagenet', include_top=False)
        print('Model loaded.')
