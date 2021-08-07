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
    cropping_layer = VideoCenterCropLayer(50, 50)
    prep_layer = BasicPreprocessing('type1', 3)
    # sample_video_path = 'datasets/datasets/Kinetic700/eval/clapping/jP4P2mHpGcU(t).mp4'
    # sample_video_path2 = 'datasets/datasets/Kinetic700/eval/playing drums/E4NJxD5tlqI(t).mp4'
    # frames, fps, num_frames, h, w = get_frames(sample_video_path)
    # frames2, fps2, num_frames2, h2, w2 = get_frames(sample_video_path2)
    # print(frames.shape)
    # print(frames.dtype)
    # print(frames2.shape)
    # print(frames2.dtype)
    records_path = 'datasets/datasets/sample_tfrecords'

    dec = TFRecord2Video()
    ds = []
    for file in [path for path in os.listdir(records_path) if path.endswith('tfrecords')]:
        ds.append(dec.get_records(os.path.join(records_path, file)))
    dataset = ds[0]
    dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=2))
    for batch in dataset:
        new = prep_layer(batch['vid'])
        print(new.shape)
    #
    # croped_frames = cropping_layer(batch)
    # print(croped_frames.shape)
    # print(type(croped_frames))

    # save_video_from_frames(frames, fps, os.path.join(PATH_TO_LOCAL_DRIVE, 'sample_for_save', 'vid.avi'))
