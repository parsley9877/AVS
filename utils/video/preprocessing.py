import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Iterable
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Rescaling,\
    CenterCrop, RandomCrop
import matplotlib.pyplot as plt

import sys
sys.path.append(os.getcwd())

from configs.global_namespace import PATH_TO_PROJECT, PATH_TO_LOCAL_DRIVE, PATH_TO_PERSISTENT_STORAGE
from utils.video.general import get_frames, save_video_from_frames
from utils.video.visualization import disp_video
from dataset_utils.loader import TFRecord2Video
from utils.video.decoder import decode_video


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
        if mode == 'type0':
            process_rescaling_layer = Rescaling(255.0 / 255.0)
            self.process_layers = Sequential([process_rescaling_layer])
        elif mode == 'type2':
            assert mean is not None and var is not None
            assert len(mean) == channel and len(var) == channel
            self.process_layers = Sequential([Rescaling(1.0 / 255.0), Normalization(mean=mean, variance=var)])
        elif mode == 'type1':
            self.process_layers = Sequential([Rescaling(1.0 / 255.0)])
        elif mode == 'type3':
            assert adaption_data is not None
            assert adaption_data.shape[-1] == channel
            self.process_layers = Sequential([Rescaling(1.0 / 255.0), Normalization().adapt(adaption_data / 255.0)])
        elif mode == 'type4':
            assert adaption_data is not None
            assert adaption_data.shape[-1] == channel
            self.process_layers = Sequential([Normalization().adapt(adaption_data)])

    def call(self, input_data: tf.Tensor, training=None, mask=None):
        """
        Description: process the input batch of data
        :param input_data: raw input batch
        :param training:
        :param mask:
        :return: processed batch
        """
        x = self.process_layers(input_data)
        return x


class VideoCenterCropLayer(keras.layers.Layer):
    def __init__(self, h: int, w: int):
        super(VideoCenterCropLayer, self).__init__()
        self.crop_layer = CenterCrop(height=h, width=w)
    def call(self, inputs, training=None, mask=None):
        return self.crop_layer(inputs)

class VideoRandomCropLayer(keras.layers.Layer):
    def __init__(self, h: int, w: int):
        super(VideoRandomCropLayer, self).__init__()
        self.crop_layer = RandomCrop(height=h, width=w)
    def call(self, inputs, training=None, mask=None):
        return self.crop_layer(inputs)
#
if __name__ == '__main__':
    prep = BasicPreprocessing('type2', 3, (0.5,0.5,0.5), (0.25,0.25,0.25))
    prep.build(input_shape=(None, 100,100,3))
    prep.summary()
