from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.optimizers import rmsprop
from scipy.misc import imsave
import os.path
import numpy as np
from ImageLoader import ImageLoader
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import applications, backend
from keras import optimizers
from keras.models import Sequential
from keras import callbacks
from keras.layers import Dropout, Flatten, Dense

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

    def __init__(self, classes, batch_size):
        """
        initialise with number of classes and path to weights file
        :param classes: the number of classes in the classification problem
        """
        self.img_dims = [244, 244]
        self.batch_size = batch_size
        self.classes = classes
        # self.weights_path = vgg16_weights_path
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
        datagen.fit(train)  # calculate stats based on the training set (should be close enough to the test set)

        return (datagen.flow(train, train_labels, batch_size=self.batch_size, shuffle=False),
                datagen.flow(test, test_labels, batch_size=self.batch_size, shuffle=False))

    def train_top_model(self, _train_labels, _test_labels, epochs):
        """
        trains the top densly connected model based on saved outputs from before the VGG16 bottleneck
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


        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:], name='flatten_01'))
        model.add(Dense(256, activation='relu', name='dense_relu_01'))
        model.add(Dropout(0.5, name='dropout_01'))
        model.add(Dense(1, activation='sigmoid', name='dense_sigmoid_02'))

        #model.summary()
        model.compile(optimizer=rmsprop(lr=0.0005),
                      loss='binary_crossentropy', metrics=['accuracy'])

        #self.sess.run(tf.global_variables_initializer())
        #file_writer = tf.summary.FileWriter(TENSORBOARD_PATH, self.sess.graph)

        model.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=self.batch_size,
                  callbacks=[self.tensorboard, history, self.checkpointer_best ],
                  validation_data=(validation_data, validation_labels))

        model.save_weights(TOPMODEL_WEIGHTS_PATH)

    def save_generator_data(self, generator, steps):
        """
        Saves the data from this generator and returns them as a numpy array of training data and labels
        :param generator: A generator object that yield training data and labels in batches
        :param steps: The number of steps (batches) to yield from the generator
        :return: numpy arrays (in a tuple) representing the training data and labels
        """
        train_set = np.empty((steps*self.batch_size, 224, 224, 3))
        label_set = np.empty((steps * self.batch_size, self.classes))

        batch_counter = 0
        global_counter = 0
        for ii, data_x in enumerate(generator):
            for imIndex in range(data_x[0].shape[0]):
                im = data_x[0][imIndex]
                label = data_x[1][imIndex]
                train_set[global_counter] = im
                label_set[global_counter] = label
                global_counter += 1
            batch_counter += 1
            if (batch_counter == steps):
                break

        return (train_set, label_set)

    def save_bottlebeck_features(self, train_data, test_data):
        """
        Feeds the data through a VGG16 model and saves the predictions for future fine-tuning
        :param train_data: a generator object based on the training data
        :param test_data: a generator object based on the testing data
        :return: None
        """
        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')
        # save the features predicted from the training set so we don't have to run the full VGG16 again
        bottleneck_features_train = model.predict(train_data, self.batch_size)
        np.save(open(BOTTLENECK_TRAIN_PATH, 'wb'), bottleneck_features_train)
        print('Bottleneck train set saved to {0}'.format(BOTTLENECK_TRAIN_PATH))
        # save the features predicted from the testing set so we don't have to run the full VGG16 again
        bottleneck_features_validation = model.predict(test_data, self.batch_size)
        np.save(open(BOTTLENECK_TEST_PATH, 'wb'), bottleneck_features_validation)
        print('Bottleneck test set saved to {0}'.format(BOTTLENECK_TEST_PATH))

    def predict_image_set(self):
        return

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
    batch_size = 32
    batches_to_generate = 100
    epochs = 10

    fruitModel = fruitCNN(1, batch_size)
    if os.path.isfile(MODEL_DATA_PATH) and os.path.isfile(MODEL_LABELS_PATH):
        fruitModel.loadDatasetFromFile(MODEL_DATA_PATH,MODEL_LABELS_PATH)
    else:
        fruitModel.addToDataset(r'D:\projects\GYA\Test_data_1\train_nongrape.txt', np.array([0]))
        fruitModel.addToDataset(r'D:\projects\GYA\Test_data_1\train_grape.txt', np.array([1]),
                                MODEL_DATA_PATH,MODEL_LABELS_PATH)


    #testImageDataFromGenerator(train_gen, 4)

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
