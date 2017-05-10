# loads image data from labelled dataset.
from scipy.misc import imsave
import numpy as np
import utils as ut
import os.path
from tqdm import trange
from tqdm import tqdm

class ImageLoader:
    """
    Loads images for training from.
    """

    def __init__(self, path, image_size=(224, 224, 3)):
        self.image_size = image_size
        rootPath = os.path.split(path)[0]
        try:
            self.logFile = open(path, 'r')
            print('\nIO: {} opened'.format(path))
        except:
            print('\nIO: {} could NOT be opened!'.format(path))
        file_lines = self.logFile.readlines()

        # first loop to find the total matches
        totalMatches = 0
        for line in file_lines:
            entries = line.split(' ')
            totalMatches += int(entries[1])

        self.train_set = np.empty((totalMatches, self.image_size[0], self.image_size[1], self.image_size[2]))
        counter = 0
        # now loop through all lines to add all matches to training tensor
        print('Processing {:d} lines containing {:d} matches...'.format(len(file_lines), totalMatches))
        for line in tqdm(file_lines, desc='Build train_set'):
            entries = line.split(' ')
            if (len(entries) < 6):
                continue
            # ignore the first two entries (image path and total entries) and the line ending
            matches = int(entries[1])
            imPath = entries[0].split('/')[1]
            imPath = os.path.join(rootPath, imPath)
            for i in range(matches):
                im = ut.load_image_without_resize(imPath)
                indices = [int(entries[i*6 + 2]), int(entries[i*6 + 4]), int(entries[i*6 + 3]), int(entries[i*6 + 5])]
                cropped = ut.crop_to_coords(im, *indices, new_size=(self.image_size[0],self.image_size[1]))
                self.train_set[counter] = cropped
                counter += 1


    def closeFile(self):
        self.logFile.close()

    def constructSampleImage(self, path, sampleWidth, sampleHeight, startIndex=0):
        """
        Constructs a summary image of the current training tensor
        :param path: the path where the image is saved to
        :param sampleWidth: the width of the image in samples (NOT pixels)
        :param sampleHeight: the height of the image in samples (NOT pixels)
        :param startIndex: where in the tensor to start drawing samples
        :return: None
        """
        # build a black picture with enough space for
        # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
        img_width = self.train_set.shape[1]
        img_height = self.train_set.shape[2]
        margin = 5
        width = sampleWidth * img_width + (sampleWidth - 1) * margin
        height = sampleHeight * img_height + (sampleHeight - 1) * margin
        stitched_samples = np.zeros((width, height, 3))

        # fill the picture with our saved filters
        print('Building {:d}x{:d} image using maximum of {:d} samples'.format(width, height, sampleWidth*sampleHeight))
        for i in trange(sampleWidth, desc='Building summary image'):
            for j in range(sampleHeight):
                imIndex = i * sampleHeight + j + startIndex
                if (imIndex >= self.train_set.shape[0]):
                    break
                img = self.train_set[imIndex]
                stitched_samples[(img_width + margin) * i: (img_width + margin) * i + img_width,
                (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

        imsave(path, stitched_samples)

if __name__ == "__main__":
    imLoader = ImageLoader(r"D:\projects\GYA\Set_1\train_grapes1.txt")
    imLoader.closeFile()
    imLoader.constructSampleImage('./grape_matches_set1.png',15,15)