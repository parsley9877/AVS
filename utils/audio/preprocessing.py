import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from python_speech_features import mfcc
from scipy.io import wavfile
import numpy
import librosa


class MFCC(keras.Model):
    """
    Description: a class for handling MFCC feature extraction
    """
    def __init__(self, specs: dict, delta: int):
        """
        Description: constructor takes a dictionary as specs of MFCC features

        :param specs: specs of MFCC algorithm. Look at the json file in this directory for more info.
        :param delta: the order of MFCC deltas needed to append (0<=delta<=2)
        """

        self.specs = specs
        self.delta = delta

    def process(self, signal: tf.Tensor):
        """
        Description: returns MFCC features of signal

        :param signal: a 1D tf.Tensor
        :return: a (:, :) or (:, :, 2) or (:, :, 3) feature MFCC feature matrix
        """
        signal_numpy = signal.numpy()
        mfcc_numpy = mfcc(signal_numpy, **self.specs)
        if self.delta == 0:
            result = mfcc_numpy
            pass
        elif self.delta == 1:
            d_mfcc_numpy = librosa.feature.delta(mfcc_numpy, order=1)
            result = np.stack([mfcc_numpy, d_mfcc_numpy], axis=-1)
        elif self.delta == 2:
            d_mfcc_numpy = librosa.feature.delta(mfcc_numpy, order=1)
            dd_mfcc_numpy = librosa.feature.delta(mfcc_numpy, order=2)
            result = np.stack([mfcc_numpy, d_mfcc_numpy, dd_mfcc_numpy], axis=-1)

        return tf.constant(result, dtype=tf.float32)

    def call(self, signal):
        """
        Description: returns MFCC features of signal

        :param signal: a 1D tf.Tensor
        :return: a (:, :) or (:, :, 2) or (:, :, 3) feature MFCC feature matrix
        """
        signal_numpy = signal.numpy()
        return tf.constant(mfcc(signal_numpy, **self.specs), dtype=tf.float32)
