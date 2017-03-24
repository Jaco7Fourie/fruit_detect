from __future__ import print_function
import sys
import threading
from multiprocessing import Queue

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import callbacks
from keras.callbacks import ModelCheckpoint
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='Run_10Feb_cifar10_withDataAug_2.txt',
                    filemode='w')


logging.info('Done with all the imports')

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('val_loss'))
        logging.info('-----------------------------------------------------------------------')
        logging.info('Epoch ' + str(epoch) + ' - Validation loss: ' + str(logs.get('val_loss')) + ' accuracy : ' + str(logs.get('val_acc')))
        logging.info('-----------------------------------------------------------------------')

    def on_batch_end(self,batch,logs={}):
        logging.info('Batch ' + str(batch) + ' - Validation loss: ' + str(logs.get('loss')) + ', validation accuracy: ' + str(logs.get('acc')))

logging.info('History class defined')

batch_size = 32
nb_classes = 10
nb_epoch = 100
data_augmentation = True
weightSavePath = '/media/vijetha/DATA/vijetha2/Documents/imageClassification_Parag/weights/'

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
logging.info('X_train shape:' + str(X_train.shape))
logging.info(str(X_train.shape[0]) + ' train samples')
logging.info(str(X_test.shape[0]) + ' test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

history = LossHistory()

logging.info('Model buidling started')

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical')

logging.info('Model buidling and compilation finished')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

checkpointer_all = ModelCheckpoint(filepath= weightSavePath + "withDataAug_weights.{epoch:02d}.hdf5", verbose=1, save_best_only=False)
checkpointer_best = ModelCheckpoint(filepath= weightSavePath + "withDataAug_bestWeights.hdf5", verbose=1, save_best_only=True)


if not data_augmentation:
    logging.info('Not using data augmentation.')
    model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch, verbose=2, show_accuracy=True, callbacks = [history, checkpointer_all, checkpointer_best],
              validation_data=(X_test, Y_test), shuffle=True)
else:
    logging.info('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        samples_per_epoch=2*batch_size,
                        nb_epoch=nb_epoch, verbose=2, show_accuracy=True, callbacks = [history],
                        validation_data=(X_test[0:100], Y_test[0:100]),
                        nb_worker=1)

    logging.info('History of the losses is: ')
    logging.info(str(history.losses))