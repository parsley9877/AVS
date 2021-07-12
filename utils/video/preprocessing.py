import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Iterable
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Rescaling
import matplotlib.pyplot as plt


class BasicPreprocessing(keras.Model):
    """
    Description: a class for preprocessing video
    it takes a set of frames (Video) tensor [B, T, H, W, C]
    """

    def __init__(self, mode: str, channel: int, mean: Iterable[float] = None, var: Iterable[float] = None, adaption_data: tf.Tensor = None) -> None:
        """
        Description: construct the preprocessing layers, and inverse processing layers based on
        type of processing.
        :param mode: mode of preprocessing
        :param channel: number of channels
        :param mean: desired mean for each channel (type2)
        :param var: desired var for each channel (type2)
        :param adaption_data: data to adapt to normalization layer (type3, type4)
        """
        super(BasicPreprocessing, self).__init__()
        possible_types = ['type0', 'type1', 'type2', 'type3', 'type4']
        assert mode.lower() in possible_types
        self.mode = mode
        self.channel = channel
        if self.mode == 'type0':
            process_rescaling_layer = Rescaling(255.0 / 255.0)
            self.process_layers = Sequential([process_rescaling_layer])
            self.inverse_process_layers = Sequential([process_rescaling_layer])
        elif self.mode == 'type2':
            assert mean is not None and var is not None
            assert len(mean) == channel and len(var) == channel
            process_normalization_layer = Normalization(mean=mean, variance=var)
            process_rescaling_layer = Rescaling(1.0 / 255.0)
            inverse_process_normalization_layer1 = Normalization(mean=[0 for x in mean],
                                                                 variance=[1.0/x for x in var])
            inverse_process_normalization_layer2 = Normalization(mean=[-x for x in mean],
                                                                 variance=[1.0/(255.0 ** 2) for x in var])
            self.inverse_process_layers = Sequential([inverse_process_normalization_layer1, inverse_process_normalization_layer2])
            self.process_layers = Sequential([process_rescaling_layer, process_normalization_layer])
        elif self.mode == 'type1':
            process_rescaling_layer = Rescaling(1.0 / 255.0)
            inverse_process_rescaling_layer = Rescaling(255.0 / 1.0)
            self.process_layers = Sequential([process_rescaling_layer])
            self.inverse_process_layers = Sequential([inverse_process_rescaling_layer])
        elif self.mode == 'type3':
            assert adaption_data is not None
            assert adaption_data.shape[-1] == channel
            process_normalization_layer = Normalization()
            process_normalization_layer.adapt(adaption_data / 255.0)
            process_rescaling_layer = Rescaling(1.0 / 255.0)
            inverse_process_normalization_layer1 = Normalization(mean=[0 for x in mean],
                                                                 variance=[1.0/x for x in process_normalization_layer.variance.numpy()])
            inverse_process_normalization_layer2 = Normalization(mean=[-x for x in process_normalization_layer.mean.numpy()],
                                                                 variance=[1.0 / (255 ** 2) for x in var])
            self.process_layers = Sequential([process_rescaling_layer, process_normalization_layer])
            self.inverse_process_layers = Sequential([inverse_process_normalization_layer1, inverse_process_normalization_layer2])
        elif self.mode == 'type4':
            assert adaption_data is not None
            assert adaption_data.shape[-1] == channel
            process_normalization_layer = Normalization()
            process_normalization_layer.adapt(adaption_data)
            inverse_process_normalization_layer1 = Normalization(mean=[0 for x in mean],
                                                                 variance=[1.0 / x for x in process_normalization_layer.variance.numpy()])
            inverse_process_normalization_layer2 = Normalization(mean=[-x for x in process_normalization_layer.mean.numpy()],
                                                                 variance=[1.0  for x in var])
            self.process_layers = Sequential([process_normalization_layer])
            self.inverse_process_layers = Sequential([inverse_process_normalization_layer1, inverse_process_normalization_layer2])

    def process(self, input_data: tf.Tensor):
        """
        Description: process the input batch of data
        :param input_data: raw input batch
        :return: processed batch
        """
        x = self.process_layers(input_data)
        return x

    def call(self, input_data: tf.Tensor):
        """
        Description: process the input batch of data
        :param input_data: raw input batch
        :return: processed batch
        """
        x = self.process_layers(input_data)
        return x

    def inverse_process(self, input_data: tf.Tensor):
        """
        Description: inverse process on processed video
        Note: reconstructed tensors are not working in tf.experimental.numpy.allclose(),
        when batch is large.
        :param input_data: processed input
        :return: reconstructed input
        """
        x = self.inverse_process_layers(input_data)
        return x
