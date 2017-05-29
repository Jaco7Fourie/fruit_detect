import numpy as np
import utils as ut
from PIL import ImageDraw, Image
import os.path
from keras.optimizers import rmsprop
from tqdm import trange
from scipy.misc import imsave
from FruitCNN import FruitCNN
from matplotlib import cm
from helper import *


def find_matches(class_model, weights_path, imPath, batch_size=32):
    """
    Adapted from http://silverpond.com.au/2016/10/24/pedestrian-detection-using-tensorflow-and-inception.html
    Takes model and an image as input as generates a grid of scores that can be used to find object matches
    :param class_model: ClassifierModel model to use as input
    :param weights_path: path to the hdf5 model weights file for the classifier head
    :param imPath: path to the input image
    :return: numpy matrix representing the grid of scores
    """
    # pre-process image
    im = ut.load_image_without_resize(imPath)
    image_data = np.empty((1, im.shape[0], im.shape[1], im.shape[2]))
    image_data[0] = im

    # run the image through a modified model
    cnn = FruitCNN(class_model, 1, batch_size)
    print('calculating bottleneck features...({0})'.format(image_data.shape[0]))
    # 2000x3000 input image results in 61x92 grid of 2048-dimensional feature vectors (using IncpetionV3 without pooling layer)
    bottleneck_features = class_model.keras_model.predict(image_data, batch_size, verbose=1)

    # 2. Add the top layer
    if not os.path.isfile(weights_path):
        print('The weights file at {0} could not be read'.format(weights_path))
        return None

    model = cnn.top_model(bottleneck_features.shape[3:])
    # don't add the rounder. We need to get uncertainties
    #model.add(Rounder())

    model.load_weights(weights_path)
    model.compile(optimizer=rmsprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    print('Model built from {0}'.format(weights_path))

    print('calculating top model features...')
    grid_height = bottleneck_features.shape[1]
    grid_width = bottleneck_features.shape[2]
    grid_results = np.ndarray((grid_height, grid_width), dtype=np.float32)
    for i in trange(grid_height, desc='grid rows'):
        for j in range(grid_width):
            feature = bottleneck_features[0, i, j]
            pred = model.predict(np.reshape(feature, (1, bottleneck_features.shape[3])),
                                 batch_size=batch_size)
            grid_results[i, j] = pred

    return grid_results


def display_grid_scores(imPath, grid_results):
    """
    creates a new image from the one provided that includes an overlay showing where in the image positive 
    classification matches come from
    :param imPath: the path to input image 
    :param grid_results: a numpy matrix representing the grid of classification results
    :return: a new image that contains the grid matches as an overlay
    """
    green = (0, 255, 0)
    red = (255, 0, 0)
    border_size_fill = 8
    border_size = 2
    threshold = 0.75
    im = ut.load_image_without_resize(imPath)
    pil_im_1 = Image.fromarray(np.uint8(im * 255))
    pil_im_2 = Image.fromarray(np.uint8(im * 255))
    draw_1 = ImageDraw.Draw(pil_im_1)
    draw_2 = ImageDraw.Draw(pil_im_2)

    grid_height, grid_width = grid_results.shape
    width_stride = (im.shape[1] // grid_width)
    height_stride = (im.shape[0] // grid_height)
    border_offset_width = (im.shape[1] - grid_width*width_stride) // 2
    border_offset_height = (im.shape[0] - grid_height * height_stride) // 2
    print('creating overlays from ({} x {}) grid'.format(grid_height, grid_width))
    for i in trange(grid_height, desc='grid rows'):
        for j in range(grid_width):
            score = grid_results[i, j]
            grid_coords_1 = [border_offset_width + j * width_stride + border_size,
                           border_offset_height + i * height_stride + border_size,
                           border_offset_width + (j + 1) * width_stride - border_size,
                           border_offset_height + (i + 1) * height_stride - border_size]

            grid_coords_2 = [border_offset_width + j * width_stride + border_size_fill,
                             border_offset_height + i * height_stride + border_size_fill,
                             border_offset_width + (j + 1) * width_stride - border_size_fill,
                             border_offset_height + (i + 1) * height_stride - border_size_fill]

            if (score > threshold):  # positive result
                draw_1.rectangle(grid_coords_1, outline=green)
            else:  # negative result
                draw_1.rectangle(grid_coords_1, outline=red)

            col = (np.asarray(cm.get_cmap('jet')(score))*255).astype(int).tolist()
            draw_2.rectangle(grid_coords_2, outline=tuple(col), fill=tuple(col))

    return np.array(pil_im_1),  np.array(pil_im_2)


###################################################################################################################
if __name__ == "__main__":
    image_path = r"D:\projects\Apple tree scanner\To label\2016_12_14_09_38_50_566_S_3_F_ON_L_ON.png"
    save_path_image = r"./images/localise_result.png"
    save_path_image_map = r"./images/localise_result_map.png"
    save_path_matrix = r"./images/localise_matrix.csv"
    model_path = r"./weights/IncpetionV3_apples_weights_0.99.hdf5"

    model = ClassifierModel('InceptionV3', input_shape=(2000, 3000, 3), pooling=None)
    grid = find_matches(model, model_path, image_path, batch_size=32)
    im1, im2 = display_grid_scores(image_path, grid)
    imsave(save_path_image, im1)
    imsave(save_path_image_map, im2)
    # save matrix as csv
    np.savetxt(save_path_matrix, grid, delimiter=",")