# loads image data from labelled dataset.
import skimage
import skimage.io
import skimage.transform
import numpy as np
import utils as ut
import os.path

class ImageLoader:
    """
    Loads images for training from.
    """

    def __init__(self, path):
        rootPath = os.path.split(path)[0]
        try:
            self.logFile = open(path, 'r')
            print('IO: {} opened'.format(path))
        except:
            print('IO: {} could NOT be opened!'.format(path))
        file_lines = self.logFile.readlines()

        self.train_set = np.empty((len(file_lines), 244, 244, 3))
        for line in file_lines:

            entries = line.split(' ')
            # ignore the first two entries (image path and total entries) and the line ending
            matches = len(entries[2:-1])//6
            imPath = entries[0].split('/')[1]
            imPath = os.path.join(rootPath, imPath)
            for i in range(matches):
                im = ut.load_image_without_resize(imPath)
                indices = [int(entries[i*6 + 2]), int(entries[i*6 + 4]), int(entries[i*6 + 3]), int(entries[i*6 + 5])]
                cropped = ut.crop_to_coords(im, *indices)


    def closeFile(self):
        self.logFile.close()