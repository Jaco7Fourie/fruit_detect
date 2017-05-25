from enum import Enum

from keras.layers import Layer
from keras import backend
from keras.applications import InceptionV3, VGG16, ResNet50

#####################################################################################################################
class ClassifierModel:

    def __init__(self, model_id, include_top=False, input_shape=(224, 224, 3), pooling='avg'):
        """
        Specifies which model to use for the classifier network
        :param model_id: options include 'VGG16', 'InceptionV3', 'ResNet50'
        :param include_top: usually left as False but can be made true to test original model classifier
        :param input_shape: image shape as (width, height, channels)
        :param pooling: optional pooling that is applied on the end of the Inception model
        """
        self.input_shape = input_shape
        # use text instead of code to prevent python from building all the models during initialization
        models_dict = dict(
            InceptionV3=r"InceptionV3(include_top=include_top, weights='imagenet', input_shape=input_shape, pooling=pooling)",
            ResNet50=r"ResNet50(include_top=include_top, weights='imagenet', input_shape=input_shape)",
            VGG16=r"VGG16(include_top=include_top, weights='imagenet', input_shape=input_shape)")
        if model_id in models_dict:
            self.keras_model = eval(models_dict[model_id])
        else:
            self.keras_model = None


#####################################################################################################################
class Rounder(Layer):

    def __init__(self, **kwargs):
        super(Rounder, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    def call(self, x, mask=None):
        x1 = backend.round(x)
        return x1