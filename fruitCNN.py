from scipy.misc import imsave
import os.path
import numpy as np
from tqdm import trange
from ImageLoader import ImageLoader
from helper import *

from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import maxnorm
from keras.models import Sequential
from keras import callbacks
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import rmsprop


#VGG_MEAN = [103.939, 116.779, 123.68]
TRAIN_TEST_SPLIT = 0.8
BOTTLENECK_TRAIN_PATH = r'./output/bottleneck_features_train.npy'
BOTTLENECK_TEST_PATH = r'./output/bottleneck_features_test.npy'
BOTTLENECK_TRAIN_LABELS_PATH = r'./output/bottleneck_labels_train.npy'
BOTTLENECK_TEST_LABELS_PATH = r'./output/bottleneck_labels_test.npy'
TOPMODEL_WEIGHTS_PATH = r'./output/bottleneck_fc_model.h5'
MODEL_DATA_PATH = r'./output/model_data.npy'
MODEL_LABELS_PATH = r'./output/model_labels.npy'
MODEL_BEST_WEIGHTS = r'./weights/checkpoints/weights.{epoch:02d}-{val_acc:.2f}.hdf5'
TENSORBOARD_PATH = r'./tensorboard'
MODEL_WEIGHTS_FOR_PREDICTION = r'D:\Source\Python\fruit_detect\weights\vgg16_weights.08-0.97.hdf5'


class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('val_loss'))
        print('-----------------------------------------------------------------------')
        print('Epoch {0} - Validation loss: {1} Accuracy : {2:.2f}'.format(str(epoch), str(logs.get('val_loss')),
                                                                           logs.get('val_acc')*100))
        print('-----------------------------------------------------------------------')

    def on_batch_end(self, batch, logs={}):
        print(' Batch {0} - Loss: {1}, Accuracy: {2}'.format(str(batch), str(logs.get('loss')),
                                                                                  str(logs.get('acc'))))

########################################################################################################################
class fruitCNN:
    """
    Loads images for training from.
    """

    def __init__(self, model, classes, batch_size):
        """
        initialise with number of classes and path to weights file
        :param model: ClassierfierModel used for the classifier network
        :param classes: the number of classes in the classification problem
        :param batch_size: the number of images to pass through the network for each gradient step
        """
        self.model = model
        self.img_dims = model.input_shape
        self.batch_size = batch_size
        self.classes = classes
        self.data = np.empty((0, self.img_dims[0], self.img_dims[1], 3))
        self.labels = np.empty((0, classes))
        self.checkpointer_best = ModelCheckpoint(filepath=MODEL_BEST_WEIGHTS, monitor='val_acc',
                                                 verbose=1,save_best_only=True)
        self.tensorboard = TensorBoard(log_dir=TENSORBOARD_PATH, histogram_freq=1,
                                  write_graph=True, write_images=False)

        #self.sess = tf.Session()
        #backend.set_session(self.sess)

    def loadDatasetFromFile (self, dataFile='', labelFile=''):
        if os.path.isfile(dataFile) and os.path.isfile(labelFile):
            self.data = np.load(open(dataFile, 'rb'))
            self.labels = np.load(open(labelFile, 'rb'))

    def addToDataset(self, path, label, dataFile='', labelFile=''):
        """
        adds a set of imge matches to train_data based on a log file of labelled matches
        :param path: the path to the matches log
        :param label: an integer np_array on-hot vector representing the label
        :param dataFile: the path to optionally save the data object to
        :param labelFile: the path to optionally save the label object to
        :return: None
        """

        imLoader = ImageLoader(path)
        if (self.data.shape[0] == 0):
            self.data = imLoader.train_set
            entries = self.data.shape[0]
            self.labels = np.vstack([label] * entries)
        else:
            self.data = np.concatenate((self.data, imLoader.train_set), 0)
            entries = imLoader.train_set.shape[0]
            self.labels = np.concatenate((self.labels, np.vstack([label] * entries)), 0)
        imLoader.closeFile()

        if dataFile != '' and labelFile != '' :
            np.save(open(dataFile, 'wb'), self.data)
            np.save(open(labelFile, 'wb'), self.labels)

    def makeTrainTestBatches(self):
        """
        returns a generator used with  model.fit_generator to generate batches of transformed images
        :return: two batch generators compatible with model.fit_generator. The first is training, the second testing. 
        """
        # shuffle the data
        p = np.random.permutation(self.data.shape[0])
        self.data = self.data[p]
        self.labels = self.labels[p]
        # Split into training / testing
        split_index = int(self.data.shape[0] * TRAIN_TEST_SPLIT)
        train = self.data[:split_index]
        test = self.data[split_index:]
        train_labels = self.labels[:split_index]
        test_labels = self.labels[split_index:]
        #testImageDataFromDataArray(test,test_labels,100)

        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.0,
            height_shift_range=0.0,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            featurewise_center=True,
            fill_mode='nearest')

        datagen.fit(train)
        gen1 = datagen.flow(train, train_labels, batch_size=self.batch_size, shuffle=False)
        gen2 = datagen.flow(test, test_labels, batch_size=self.batch_size, shuffle=False)
        return (gen1, gen2)



    def train_top_model(self, _train_labels, _test_labels, epochs):
        """
        trains the top densely connected model based on saved outputs from before the classifier bottleneck
        :param _train_labels: labels used in training set
        :param _test_labels: labels used in test/validation set
        :param epochs: numbers of times to go through entire dataset
        :return: None
        """
        history = LossHistory()

        train_data = np.load(open(BOTTLENECK_TRAIN_PATH, 'rb'))
        train_labels = _train_labels[:train_data.shape[0]]
        # ensure labels are in the right shape
        train_labels = np.reshape(train_labels, (train_labels.shape[0], self.classes))

        validation_data = np.load(open(BOTTLENECK_TEST_PATH, 'rb'))
        validation_labels = _test_labels[:validation_data.shape[0]]
        # ensure labels are in the right shape
        validation_labels = np.reshape(validation_labels, (validation_labels.shape[0], self.classes))

        model = self.top_model(train_data.shape[1:])

        #model.summary()
        model.compile(optimizer=rmsprop(lr=0.001),
                      loss='binary_crossentropy', metrics=['accuracy'])

        #self.sess.run(tf.global_variables_initializer())
        #file_writer = tf.summary.FileWriter(TENSORBOARD_PATH, self.sess.graph)

        model.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=self.batch_size,
                  callbacks=[self.tensorboard, history, self.checkpointer_best ],
                  validation_data=(validation_data, validation_labels))

        model.save_weights(TOPMODEL_WEIGHTS_PATH)

    def top_model(self, shape):
        """
        defines the top model of our classifier
        :param shape: the shape of the tensor to use as model input
        :return: the model
        """
        model = Sequential()
        model.add(Flatten(input_shape=shape, name='flatten_01'))
        model.add(Dense(128, activation='relu', name='dense_relu_01')) # kernel_constraint=maxnorm(4))
        model.add(Dropout(0.5, name='dropout_01'))
        model.add(Dense(1, activation='sigmoid', name='dense_sigmoid_03'))
        return model

    def save_generator_data(self, generator, steps):
        """
        Saves the data from this generator and returns them as a numpy array of training data and labels
        :param generator: A generator object that yield training data and labels in batches
        :param steps: The number of steps (batches) to yield from the generator
        :return: numpy arrays (in a tuple) representing the training data and labels
        """
        train_set = np.empty((int(steps*self.batch_size), self.img_dims[0], self.img_dims[0], 3))
        label_set = np.empty((int(steps * self.batch_size), self.classes))

        batch_counter = 0
        global_counter = 0
        for ii, data_x in enumerate(generator):
            for imIndex in range(data_x[0].shape[0]):
                im = data_x[0][imIndex]
                label = data_x[1][imIndex]
                train_set[global_counter] = im
                label_set[global_counter] = label[0]
                if (label > 1):
                    print('aah!!!')
                global_counter += 1
            batch_counter += 1
            if (batch_counter == steps):
                break

        return (train_set, label_set)

    # noinspection PyTypeChecker
    def save_bottlebeck_features(self, train_data, test_data):
        """
        Feeds the data through a classifier model and saves the predictions for future fine-tuning
        :param train_data: a generator object based on the training data
        :param test_data: a generator object based on the testing data
        :return: None
        """
        # build the classifier network
        #model = applications.VGG16(include_top=False, weights='imagenet')
        # save the features predicted from the training set so we don't have to run the full classifier again
        bottleneck_features_train = self.model.keras_model.predict(train_data, self.batch_size)
        np.save(open(BOTTLENECK_TRAIN_PATH, 'wb'), bottleneck_features_train)
        print('Bottleneck train set saved to {0}'.format(BOTTLENECK_TRAIN_PATH))
        # save the features predicted from the testing set so we don't have to run the full classifier again
        bottleneck_features_validation = self.model.keras_model.predict(test_data, self.batch_size)
        np.save(open(BOTTLENECK_TEST_PATH, 'wb'), bottleneck_features_validation)
        print('Bottleneck test set saved to {0}'.format(BOTTLENECK_TEST_PATH))

    def predict_image_set(self, image_data, labels=None):
        '''
        evaluates the model based on the image data provided and prints a summary of results
        :param image_data: image data formatted  as a tensor
        :param labels: labels can optionally be provided to evaluate accuracy
        :return: a tuple consisting of the overall accuracy followed by the model predictions 
        '''

        # 1. first feed images through to bottleneck of the classifier
        # save the features predicted from the training set so we don't have to run the full classifier again
        print('calculating bottleneck features...({0})'.format(image_data.shape[0]))
        bottleneck_features= self.model.predict(image_data, self.batch_size)

        # 2. Add the top layer
        if not os.path.isfile(MODEL_WEIGHTS_FOR_PREDICTION):
            print('The weights file at {0} could not be read'.format(MODEL_WEIGHTS_FOR_PREDICTION))
            return None

        model = self.top_model(bottleneck_features.shape[1:])
        model.add(Rounder())

        model.load_weights(MODEL_WEIGHTS_FOR_PREDICTION)
        model.compile(optimizer=rmsprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        print('Model built from {0}'.format(MODEL_WEIGHTS_FOR_PREDICTION))

        print('calculating top model features...')
        if (labels != None):
            scores = model.evaluate(bottleneck_features, labels, verbose=0)
        else:
            scores = [0,0]
        predictions = model.predict(bottleneck_features, batch_size=self.batch_size)

        return scores[1]*100, predictions

    def evaluate_image_path(self, log_paths, labels=None, images_in_mosaic=[10,10]):
        """
        evaluates the images specified in the log file on the trained model
        generated two image mosaics as output, one for positive matches and one for negatives
        :param log_paths:  a list of paths to the image log file(s)
        :param labels:  labels can optionally be associated with each path. 
        If provided it has to be a list of the same length as log_paths
        :param images_in_mosaic: the path to the image log file
        :return: None
        """
        if (labels != None and len(log_paths) != len(labels)):
            print('log_paths and labels length do not match! aborting...')
            return None

        imLoader = ImageLoader(log_paths[0])
        data = imLoader.train_set
        entries = imLoader.train_set.shape[0]
        data_labels =  np.vstack([labels[0]] * entries)
        for idx,path in enumerate(log_paths[1:], start=1):
            imLoader = ImageLoader(path)
            data = np.concatenate((data, imLoader.train_set), 0)
            entries = imLoader.train_set.shape[0]
            if (labels != None):
                data_labels = np.concatenate((data_labels, np.vstack([labels[idx]] * entries)), 0)
            else:
                data_labels = None

        # shuffle the data
        p = np.random.permutation(data.shape[0])
        data = data[p]
        data_labels = data_labels[p]
        score, predictions = self.predict_image_set(data, data_labels)
        print('Mean accuracy from labelled test set: {0}'.format(score))

        # build a black picture with enough space for
        # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
        img_width = data.shape[2]
        img_height = data.shape[1]
        margin = 5
        height = images_in_mosaic[0] * img_width + (images_in_mosaic[0] - 1) * margin
        width = images_in_mosaic[1] * img_height + (images_in_mosaic[1] - 1) * margin
        positive_samples = np.zeros((height, width, 3))
        negative_samples = np.zeros((height, width, 3))

        # fill the picture with our saved filters
        print('Building {:d}x{:d} image using maximum of {:d} samples'
              .format(width, height, images_in_mosaic[0] * images_in_mosaic[1]))

        positive_counter = 0
        negative_counter = 0
        total_images = images_in_mosaic[0] * images_in_mosaic[1]
        for i in trange(data.shape[0]):
            match = predictions[i]
            img = data[i]
            if match > 0.5 and positive_counter < total_images:
                row = positive_counter // images_in_mosaic[1]
                col = positive_counter % images_in_mosaic[1]
                positive_samples[(img_height + margin) * row: (img_height + margin) * row + img_height,
                (img_width + margin) * col: (img_width + margin) * col + img_width, :] = img
                positive_counter += 1
            elif match <= 0.5 and negative_counter < total_images:
                row = negative_counter // images_in_mosaic[1]
                col = negative_counter % images_in_mosaic[1]
                negative_samples[(img_height + margin) * row: (img_height + margin) * row + img_height,
                (img_width + margin) * col: (img_width + margin) * col + img_width, :] = img
                negative_counter += 1

        p_path = r'.\images\positive_matches.png'
        n_path = r'.\images\negative_matches.png'
        imsave(p_path, positive_samples)
        imsave(n_path, negative_samples)

########################################################################################################################


def testImageDataFromGenerator(generator, batches):
    root_path = r'.\images\test'
    counter = 0
    for ii,data_x in enumerate(generator):
        for imIndex in range(data_x[0].shape[0]):
            im = data_x[0][imIndex]
            label = data_x[1][imIndex]
            if (label == 1):
                path = os.path.join(root_path, 'grape_{0}_{1}.png'.format(ii,imIndex))
                imsave(path,im)
            else:
                path = os.path.join(root_path, 'nongrape_{0}_{1}.png'.format(ii,imIndex))
                imsave(path, im)
        counter += 1
        if (counter == batches):
            break

def testImageDataFromDataArray(data, labels, stopat=50):
    root_path = r'.\images\test'
    counter = 0
    for ii in range(np.min([data.shape[0],stopat])):
        im = data[ii]
        if (labels[ii] == 1):
            path = os.path.join(root_path, 'grape_{0}.png'.format(ii))
            imsave(path, im)
        else:
            path = os.path.join(root_path, 'nongrape_{0}.png'.format(ii))
            imsave(path, im)

########################################################################################################################
if __name__ == "__main__":
    batch_size = 64
    batches_to_generate = 100
    epochs = 15
    training = True
    model = ClassifierModel('VGG16', input_shape=(224,224,3))

    fruitModel = fruitCNN(model, 1, batch_size)
    if (training):
        if os.path.isfile(MODEL_DATA_PATH) and os.path.isfile(MODEL_LABELS_PATH):
            fruitModel.loadDatasetFromFile(MODEL_DATA_PATH,MODEL_LABELS_PATH)
        else:
            fruitModel.addToDataset(r'D:\projects\GYA\Test_data_1\train_nongrape.txt', np.array([0]))
            fruitModel.addToDataset(r'D:\projects\GYA\Test_data_1\train_grape.txt', np.array([1]),
                                    MODEL_DATA_PATH,MODEL_LABELS_PATH)


        if os.path.isfile(BOTTLENECK_TRAIN_PATH) and\
                os.path.isfile(BOTTLENECK_TEST_PATH) and\
                os.path.isfile(BOTTLENECK_TRAIN_LABELS_PATH) and\
                BOTTLENECK_TEST_LABELS_PATH:
            train_labels = np.load(open(BOTTLENECK_TRAIN_LABELS_PATH, 'rb'))
            test_labels = np.load(open(BOTTLENECK_TEST_LABELS_PATH, 'rb'))
            fruitModel.train_top_model(train_labels, test_labels, epochs)
        else:
            train_gen, test_gen = fruitModel.makeTrainTestBatches()
            train_data, train_labels = fruitModel.save_generator_data(train_gen, batches_to_generate)
            test_data, test_labels = fruitModel.save_generator_data(test_gen, np.floor(batches_to_generate*(1-TRAIN_TEST_SPLIT)))

            fruitModel.save_bottlebeck_features(train_data, test_data)
            # save the labels
            np.save(open(BOTTLENECK_TRAIN_LABELS_PATH, 'wb'), train_labels)
            np.save(open(BOTTLENECK_TEST_LABELS_PATH, 'wb'), test_labels)
    else:
        paths = [r"D:\projects\GYA\Test_data_2\train_grapes.txt", r"D:\projects\GYA\Test_data_2\train_non_grapes.txt"]
        labels = [1,0]
        fruitModel.evaluate_image_path(paths, labels, images_in_mosaic=[6,6])
